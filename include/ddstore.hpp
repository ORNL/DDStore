#include <iostream>
#include <map>
#include <mpi.h>
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <typeinfo>
#include <vector>
#include <assert.h>
#include <cstring>

#define Q_NAME "/ddstore"
#define Q_OFLAGS_CONSUMER (O_RDONLY)
#define Q_OFLAGS_PRODUCER (O_CREAT | O_WRONLY)
#define Q_MODE (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)
#define Q_ATTR_FLAGS 0
#define Q_ATTR_MSG_SIZE 4096 // 8192
#define Q_ATTR_MAX_MSG 10
#define Q_ATTR_CURMSGS 0
#define Q_CREATE_WAIT_US 1000000
#define MSG_COUNT_DEFAULT 20
#define MSG_PERIOD_US 100000

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

struct QueInfo
{
    mqd_t mqd; // data mq
    mqd_t mqr; // request mq
    std::string mqd_name;
    std::string mqr_name;
    long mqd_msgsize;
    long mqr_msgsize;
};
typedef struct QueInfo QueInfo_t;

int sortedsearch(std::vector<int> &vec, long unsigned int idx);

class DDStore
{
  public:
    DDStore();
    DDStore(MPI_Comm comm);
    DDStore(MPI_Comm comm, int use_mq, int role);
    ~DDStore();

    int use_mq;
    int role; // 0 = producer, 1 = consumer

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
        std::cout << this->rank << ": " << "l_ntotal,disp,ncount = " << l_ntotal << ", " << disp << ", " << ncount << std::endl;

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
            std::cout << this->rank << ": " << "ntotal = " << ntotal << std::endl;

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
            mq = queinfo.mqd;

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
    int get(std::string name, long unsigned int id, T *buffer, int size)
    {
        QueInfo_t queinfo;
        mqd_t mqd = 0;
        mqd_t mqr = 0;

        if (this->use_mq)
        {
            queinfo = this->qlist[name];
            mqr = queinfo.mqr;
            mqd = queinfo.mqd;
        }

        VarInfo_t varinfo = this->varlist[name];
        if (varinfo.typeinfo != typeid(T).name())
            throw std::invalid_argument("Invalid data type");
        int len = 0;
        int nbyte = 0;

        len = varinfo.lenlist[id];
        nbyte = varinfo.disp * sizeof(T) * len;
        if ((long unsigned int)nbyte > size * sizeof(T))
            throw std::invalid_argument("Invalid buffer size");

        if (this->use_mq && (this->role == 1))
        {

            // printf("[%d:%d] push request: %ld\n", this->role, this->rank, id);
            this->pushr(mqr, (char *)&id, sizeof(long unsigned int));

            // printf("[%d:%d] pull data: %d bytes\n", this->role, this->rank, nbyte);
            this->pulld(mqd, (char *)buffer, nbyte);
        }
        else
        {
            if (this->use_mq && (this->role == 0))
            {
                // get id from mqr
                this->pullr(mqr, (char *)&id, sizeof(long unsigned int));
                // printf("[%d:%d] pull request: %ld\n", this->role, this->rank, id);

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
                // printf("[%d:%d] push data: %ld\n", this->role, this->rank, id);
                this->pushd(mqd, (char *)buffer, nbyte);
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

    std::map<std::string, VarInfo_t> varlist;
    std::map<std::string, QueInfo_t> qlist;

    void queue_init(std::string name);
    void pushr(mqd_t mq, char *buffer, long size);
    void pullr(mqd_t mq, char *buffer, long size);
    void pushd(mqd_t mq, char *buffer, long size);
    void pulld(mqd_t mq, char *buffer, long size);
};
