#include <assert.h>
#include <cstring>
#include <iostream>
#include <map>
#include <mpi.h>
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <vector>

#include <unistd.h>
#include <pthread.h>
#include <sys/syscall.h>

#include <fcntl.h>
#include <errno.h>

#include <sys/mman.h>
#include <semaphore.h>

#define Q_NAME "/ddstore"
#define Q_OFLAGS_CONSUMER (O_RDONLY)
#define Q_OFLAGS_PRODUCER (O_CREAT | O_WRONLY)
#define Q_MODE (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
#define Q_ATTR_FLAGS 0
#define Q_ATTR_CURMSGS 0
#define Q_CREATE_WAIT_US 1000000
#define MSG_COUNT_DEFAULT 20
#define MSG_PERIOD_US 1000
#define NCH 4
#define SHM_QUEUE_CAPACITY 4
#define SHM_QUEUE_BUFFERSIZE 16384

struct Request
{
    long unsigned int id;
    int ich;
};
typedef struct Request Request_t;

struct VarInfo
{
    std::string name;
    std::string typeinfo;
    int itemsize;
    int disp;
    std::vector<int> lenlist;                   // list of sizes of all element
    std::vector<long unsigned int> dataoffsets; // list of offsets of all elements
    std::vector<int> offsets;                   // list of offsets at each process
    // std::vector<int> offsetlist; // list of offsets at each process
    // std::vector<int> counts;  // list of total number of elements each process has
    // std::vector<int> process_offsets; // list of offsets at each process
    MPI_Win win;
    bool active;
    bool fence_active;
    void *base;
};
typedef struct VarInfo VarInfo_t;

struct SharedQueue
{
    int head;
    int tail;
    int capacity;
    int buffersize;
    char* buffer[SHM_QUEUE_CAPACITY*SHM_QUEUE_BUFFERSIZE];
};
typedef struct SharedQueue SharedQueue_t;

struct QueInfo
{
    mqd_t mqd[64]; // data mq
    mqd_t mqr; // request mq
    std::string mqd_name[64];
    std::string mqr_name;
    long mqd_msgsize;
    long mqr_msgsize;

    SharedQueue* shqueue[64];
    int shm_fd[64];
    size_t shm_len[64];
    sem_t *sem_mutex[64];
    sem_t *sem_readable[64];
    sem_t *sem_writable[64];
};
typedef struct QueInfo QueInfo_t;

int sortedsearch(std::vector<int> &vec, long unsigned int idx);

static void enqueue(SharedQueue *queue, void *buffer, int len, sem_t *sem_mutex, sem_t *sem_readable, sem_t *sem_writable)
{
    sem_wait(sem_writable); // Wait for space to become available (blocks if the queue is full)
    sem_wait(sem_mutex); // Enter critical section

    // FIXME: warning of using pointer of type 'void *' in arithmetic
    void *ptr = (void*) queue->buffer + queue->tail * queue->buffersize;
    memcpy(ptr, buffer, len);
    queue->tail = (queue->tail + 1) % queue->capacity;

    sem_post(sem_mutex); // Leave critical section
    sem_post(sem_readable); // Signal that an item has been added
}

static void dequeue(SharedQueue *queue, void *buffer, int len, sem_t *sem_mutex, sem_t *sem_readable, sem_t *sem_writable)
{
    sem_wait(sem_readable); // Wait for an item to become available
    sem_wait(sem_mutex); // Enter critical section

    void *ptr = (void*) queue->buffer + queue->head * queue->buffersize;
    memcpy(buffer, ptr, len);
    queue->head = (queue->head + 1) % queue->capacity;

    sem_post(sem_mutex); // Leave critical section
    sem_post(sem_writable); // Signal that a space has been freed up
}

class DDStore
{
  public:
    DDStore();
    DDStore(MPI_Comm comm);
    DDStore(MPI_Comm comm, int use_mq, int role);
    DDStore(MPI_Comm comm, int use_mq, int role, int mode);
    ~DDStore();

    int use_mq;
    int role;    // 0: producer, 1: consumer
    int mode;    // 0: mq, 1: stream mq, 2: shmem
    int verbose; // 0: non verbose, 1: verbose
    int ndchannel; // number of data channels
    long attr_msg_size;
    long attr_max_msg;

    void query(std::string name, VarInfo_t &varinfo);
    void epoch_begin();
    void epoch_end();
    void free();

    /*
    ** Data layout sketch:
    ** We assume an array of variable-length data per process.
    ** Each element in the array is a block of n x disp elements (n: len, disp:
    *displacement)
    ** E.g.,
    **      +-------+-------+-------+
    **      |<- 1 ->|<- 2 ->|<- 3 ->|
    **      +-------+-------+-------+
    ** len is width (variable)
    ** disp is height (same for all)
    */

    template <typename T>
    void create(std::string name, T *buffer, int disp, int *l_lenlist, int ncount)
    {
        long l_ntotal = 0;
        for (int i = 0; i < ncount; i++)
            l_ntotal += l_lenlist[i];
        std::cout << this->rank << ": "
                  << "l_ntotal,disp,ncount = " << l_ntotal << ", " << disp << ", " << ncount << std::endl;

        VarInfo_t var;
        MPI_Win win;
        void *base = NULL;

        if (!this->use_mq || (this->use_mq && (this->role == 0)))
        {
            int err = MPI_Alloc_mem((MPI_Aint)(l_ntotal * disp * sizeof(T)), MPI_INFO_NULL, &base);
            if (err)
            {
                exit(1);
            }
            std::memcpy(base, buffer, l_ntotal * disp * sizeof(T));

            MPI_Win_create(base,                                  /* pre-allocated buffer */
                           (MPI_Aint)l_ntotal * disp * sizeof(T), /* size in bytes */
                           disp * sizeof(T),                      /* displacement units */
                           MPI_INFO_NULL,                         /* info object */
                           this->comm,                            /* communicator */
                           &win);                                 /* window object */

            // Use MPI_Allgatherv
            // Note: expect possible limit for using int32 (due to MPI)
            std::vector<int> recvcounts(this->comm_size);
            std::vector<int> displs(this->comm_size);

            MPI_Allgather(&ncount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, this->comm);
            int ntotal = 0;
            {
                long unsigned int i = 0;
                for (auto &x : recvcounts)
                {
                    displs[i++] = ntotal;
                    ntotal += x;
                }
            }

            // for (long unsigned int i = 0; i < recvcounts.size(); i++)
            //     std::cout << this->rank << ": " << "recvcounts[" << i << "] = " << recvcounts[i] << std::endl;
            // for (long unsigned int i = 0; i < displs.size(); i++)
            //     std::cout << this->rank << ": " << "displs[" << i << "] = " << displs[i] << std::endl;
            std::cout << this->rank << ": "
                      << "ntotal = " << ntotal << std::endl;

            std::vector<int> lenlist(ntotal);
            std::vector<long unsigned int> dataoffsets(ntotal);
            /*
            int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype
                                sendtype, void *recvbuf, const int *recvcounts, const int *displs,
                                MPI_Datatype recvtype, MPI_Comm comm)
            */
            MPI_Allgatherv(l_lenlist, ncount, MPI_INT, lenlist.data(), recvcounts.data(), displs.data(), MPI_INT, this->comm);
            {
                long unsigned int sum = 0;
                long unsigned int i = 0;
                for (auto &x : lenlist)
                {
                    dataoffsets[i++] = sum;
                    sum += x;
                }
            }

            // for (long unsigned int i = 0; i < lenlist.size(); i++)
            //     std::cout << this->rank << ": " << "lenlist[" << i << "] = " << lenlist[i] << std::endl;
            // for (long unsigned int i = 0; i < dataoffsets.size(); i++)
            //     std::cout << this->rank << ": " << "dataoffsets[" << i << "] = " << dataoffsets[i] << std::endl;

            int max_disp = 0;
            // We assume disp is same for all
            MPI_Allreduce(&disp, &max_disp, 1, MPI_INT, MPI_MAX, this->comm);
            if (max_disp != disp)
                throw std::invalid_argument("Invalid disp");

            var.lenlist = lenlist;
            var.dataoffsets = dataoffsets;
            var.offsets = displs;
        }

        // std::cout << this->rank << ": " << "use_mq,role = " << this->use_mq << ", " << this->role << std::endl;
        if (this->use_mq)
        {
            queue_init(name);
        }

        // exchange lenlist
        if (this->use_mq)
        {
            QueInfo_t queinfo;
            mqd_t mq = 0;

            queinfo = this->qlist[name];
            mq = queinfo.mqd[0];

            int rc = 0;
            int ntotal = 0;
            int nbytes = 0;
            struct mq_attr attr;
            mq_getattr(mq, &attr);

            int nchunk;

            if (this->role == 0)
            {
                ntotal = var.lenlist.size();
                rc = mq_send(mq, (char *)&ntotal, sizeof(int), 0);

                nchunk = ntotal * sizeof(int) / attr.mq_msgsize;
                if ((ntotal * sizeof(int)) % attr.mq_msgsize != 0)
                    nchunk += 1;
                // printf("[%d:%d] pushd: send ntotal (%d): %d %ld %d\n", this->role, this->rank, rc, ntotal, ntotal * sizeof(int), nchunk);
                for (int i = 0; i < nchunk; i++)
                {
                    int len = attr.mq_msgsize;
                    if (i == nchunk - 1)
                        len = ntotal * sizeof(int) - i * attr.mq_msgsize;

                    // printf("[%d:%d] pushd: ready to send: %d %d\n", this->role, this->rank, i, len);
                    rc = mq_send(mq, (char *)var.lenlist.data() + i * attr.mq_msgsize, len, 0);
                    if (rc < 0)
                    {
                        printf("[%d:%d] Error send lenlist (%d), i,total: %d %d\n", this->role, this->rank, rc, i, nbytes);
                        perror("producer error on mq_send: ");
                        i--;
                        continue;
                    }
                    nbytes += len;
                    // printf("[%d:%d] pushd: send lenlist (%d), i,total: %d %d\n", this->role, this->rank, rc, i, nbytes);
                }

                // for (int i = 0; i < ntotal; i++)
                // {
                //     printf("[%d:%d] lenlist[%d]: %d\n", this->role, this->rank, i, var.lenlist[i]);
                // }
            }
            else
            {
                rc = mq_receive(mq, (char *)&ntotal, attr.mq_msgsize, NULL);

                std::vector<int> lenlist(ntotal);
                nchunk = ntotal * sizeof(int) / attr.mq_msgsize;
                if ((ntotal * sizeof(int)) % attr.mq_msgsize != 0)
                    nchunk += 1;
                // printf("[%d:%d] pulld: recv ntotal (%d): %d %ld %d\n", this->role, this->rank, rc, ntotal, ntotal * sizeof(int), nchunk);

                for (int i = 0; i < nchunk; i++)
                {
                    rc = mq_receive(mq, (char *)lenlist.data() + i * attr.mq_msgsize, attr.mq_msgsize, NULL);
                    if (rc < 0)
                    {
                        printf("[%d:%d] Error recv lenlist (%d), i,total: %d %d\n", this->role, this->rank, rc, i, nbytes);
                        perror("consumer error on mq_receive: ");
                        i--;
                        continue;
                    }
                    nbytes += rc;
                    // printf("[%d:%d] pulld: recv lenlist (%d), i,total: %d %d\n", this->role, this->rank, rc, i, nbytes);
                }

                var.lenlist = lenlist;

                // for (int i = 0; i < ntotal; i++)
                // {
                //     printf("[%d:%d] lenlist[%d]: %d\n", this->role, this->rank, i, var.lenlist[i]);
                // }
            }
        }

        var.name = name;
        var.typeinfo = typeid(T).name();
        var.itemsize = sizeof(T);
        var.disp = disp;
        var.win = win;
        var.active = true;
        var.fence_active = false;
        var.base = base;

        this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
    }

    template <typename T>
    int get(std::string name, long unsigned int id, T *buffer, int size, int stream_ichannel = 0)
    {
        QueInfo_t queinfo;
        mqd_t mqd = 0;
        mqd_t mqr = 0;
        Request_t req;
        int ich;

        SharedQueue* shqueue;
        sem_t *sem_mutex;
        sem_t *sem_readable;
        sem_t *sem_writable;

        // pid_t pid = getpid();
        pid_t tid = syscall(SYS_gettid);
        // pthread_t tid2 = pthread_self();

        if (this->use_mq && (this->role == 1))
        {
            if (this->channelmap.find(tid) == this->channelmap.end())
            {
                /*
                // multi-thread
                pthread_spin_lock(&(this->spinlock));
                assert(this->imax < this->ndchannel);
                printf("[%d:%d:%d] insert channelmap: %d\n", this->role, this->rank, tid, this->imax);
                this->channelmap.insert(std::pair<pid_t, int>(tid, this->imax));
                this->imax = this->imax + 1;
                pthread_spin_unlock(&(this->spinlock));
                */

                // multi-process
                // Initializing the flock structure
                struct flock lock;
                lock.l_type = F_WRLCK; // Exclusive write lock
                lock.l_whence = SEEK_SET; // Relative to the beginning of the file
                lock.l_start = 0; // Start of the lock
                lock.l_len = 0; // 0 means lock the whole file

                int fd;
                char fname[128];
                snprintf(fname, 128, "ddstore-filelock-%d.lock", this->rank);
                fd = open(fname, O_RDWR);

                // Attempting to acquire the lock
                if (fcntl(fd, F_SETLKW, &lock) == -1)
                {
                    perror("Error locking file");
                }

                char buffer[4];
                int bytesRead;
                bytesRead = read(fd, buffer, 3);
                buffer[bytesRead] = '\0';
                this->imax = atoi(buffer);

                // insert
                this->channelmap.insert(std::pair<pid_t, int>(tid, this->imax));
                // printf("[%d:%d:%d] insert channelmap: %d\n", this->role, this->rank, tid, this->imax);
                this->imax = this->imax + 1;

                // Now the file is locked, we can write to it
                snprintf(buffer, 4, "%3d", this->imax);
                lseek(fd, 0, SEEK_SET);
                write(fd, buffer, 3);

                // Unlocking the file
                lock.l_type = F_UNLCK;
                if (fcntl(fd, F_SETLK, &lock) == -1)
                {
                    perror("Error unlocking file");
                }
                close(fd);
            }
        }

        if (this->use_mq)
        {
            queinfo = this->qlist[name];
            mqr = queinfo.mqr;
        }

        VarInfo_t varinfo = this->varlist[name];
        if (varinfo.typeinfo != typeid(T).name())
            throw std::invalid_argument("Invalid data type");
        int len = 0;
        int nbyte = 0;

        len = varinfo.lenlist[id];
        nbyte = varinfo.disp * sizeof(T) * len;
        if (this->use_mq && (this->role == 1))
        {
            if ((long unsigned int)nbyte > size * sizeof(T))
                throw std::invalid_argument("Invalid buffer size");
        }

        if (this->use_mq && (this->role == 1))
        {
            ich = this->channelmap[tid];
            if (this->verbose)
                printf("[%d:%d:%d] ich: %d\n", this->role, this->rank, tid, ich);
            mqd = queinfo.mqd[ich];

            shqueue = queinfo.shqueue[ich];
            sem_mutex = queinfo.sem_mutex[ich];
            sem_readable = queinfo.sem_readable[ich];
            sem_writable = queinfo.sem_writable[ich];

            if ((this->mode == 0) || (this->mode == 2))
            {
                if (this->verbose)
                    printf("[%d:%d:%d] push request: %ld %d\n", this->role, this->rank, tid, id, ich);
                req.id = id;
                req.ich = ich;
                this->pushr(mqr, (char *)&req, sizeof(Request_t));
            }

            if (this->verbose)
                printf("[%d:%d:%d] pull data: %d bytes\n", this->role, this->rank, tid, nbyte);

            // we assume the buffer is always big enough for stream get
            if (this->mode < 2)
            {
                this->pulld(mqd, (char *)buffer, nbyte);
            }
            else
            {
                this->pulld(shqueue, (char *)buffer, nbyte, sem_mutex, sem_readable, sem_writable);
            }
        }
        else
        {
            if (this->use_mq && (this->role == 0))
            {
                if ((this->mode == 0) || (this->mode == 2))
                {
                    // get id from mqr
                    this->pullr(mqr, (char *)&req, sizeof(Request_t));
                    id = req.id;
                    ich = req.ich;
                    mqd = queinfo.mqd[ich];

                    shqueue = queinfo.shqueue[ich];
                    sem_mutex = queinfo.sem_mutex[ich];
                    sem_readable = queinfo.sem_readable[ich];
                    sem_writable = queinfo.sem_writable[ich];

                    if (this->verbose)
                        printf("[%d:%d] pull request: %ld %d\n", this->role, this->rank, id, ich);
                }
                else
                {
                    // stream mode
                    mqd = queinfo.mqd[stream_ichannel];
                    if (this->verbose)
                        printf("[%d:%d] stream: %ld %d\n", this->role, this->rank, id, stream_ichannel);
                }

                // reset based on the requested id
                len = varinfo.lenlist[id];
                nbyte = varinfo.disp * sizeof(T) * len;
                // buffer = (T *)std::malloc(nbyte);
                MPI_Alloc_mem((MPI_Aint)nbyte, MPI_INFO_NULL, &buffer);
            }

            int target = sortedsearch(varinfo.offsets, id);
            long unsigned int offset = varinfo.dataoffsets[varinfo.offsets[target]];
            long unsigned int dataoffset = varinfo.dataoffsets[id];
            // std::cout << "[" << this->rank << "] id,target,offset,dataoffset,len: " << id << "," << target << "," << offset << "," << dataoffset << "," << len << std::endl;

            MPI_Win win = varinfo.win;
            // printf("[%d:%d] MPI_Win_lock\n", this->role, this->rank);
            MPI_Win_lock(MPI_LOCK_SHARED, target, 0, win);
            /*
            int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype
                        origin_datatype, int target_rank, MPI_Aint target_disp,
                        int target_count, MPI_Datatype target_datatype, MPI_Win
                        win)
            */
            // printf("[%d:%d] MPI_Get\n", this->role, this->rank);
            MPI_Get(buffer,              /* pre-allocated buffer on RMA origin process */
                    nbyte,               /* count on RMA origin process */
                    MPI_BYTE,            /* type on RMA origin process */
                    target,              /* rank of RMA target process */
                    dataoffset - offset, /* displacement on RMA target process */
                    nbyte,               /* count on RMA target process */
                    MPI_BYTE,            /* type on RMA target process */
                    win);                /* window object */
            // printf("[%d:%d] MPI_Win_unlock\n", this->role, this->rank);
            MPI_Win_unlock(target, win);

            if (this->use_mq && (this->role == 0))
            {
                if (this->verbose)
                    printf("[%d:%d] push data: %ld\n", this->role, this->rank, id);
                
                if (this->mode < 2)
                {
                    this->pushd(mqd, (char *)buffer, nbyte);
                }
                else
                {
                    this->pushd(shqueue, (char *)buffer, nbyte, sem_mutex, sem_readable, sem_writable);
                }

                // std::free((void *)buffer);
                MPI_Free_mem((void *)buffer);
            }
        }

        return 0;
    }

  private:
    MPI_Comm comm;
    int comm_size;
    int rank;
    pthread_spinlock_t spinlock;

    std::map<std::string, VarInfo_t> varlist;
    std::map<std::string, QueInfo_t> qlist;
    std::map<pid_t, int> channelmap;
    int imax;

    void queue_init(std::string name);
    void pushr(mqd_t mq, char *buffer, long size);
    void pullr(mqd_t mq, char *buffer, long size);
    void pushd(mqd_t mq, char *buffer, long size);
    void pulld(mqd_t mq, char *buffer, long size);
    void pushd(SharedQueue *queue, char *buffer, long size, sem_t *sem_mutex, sem_t *sem_readable, sem_t *sem_writable);
    void pulld(SharedQueue *queue, char *buffer, long size, sem_t *sem_mutex, sem_t *sem_readable, sem_t *sem_writable);
};
