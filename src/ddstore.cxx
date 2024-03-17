#include "ddstore.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef USE_BOOST
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup.hpp>

#define LOG BOOST_LOG_TRIVIAL(debug)
#endif

int sortedsearch(std::vector<int> &vec, long unsigned int idx)
{
    int rtn = 0;
    for (long unsigned int i = 0; i < vec.size() - 1; i++)
    {
        if ((idx >= vec[i]) && (idx < vec[i + 1]))
        {
            rtn = i;
            break;
        }
    }

    if (idx >= vec[vec.size() - 1])
        rtn = vec.size() - 1;

    if (idx < vec[0])
        rtn = -1;

    return rtn;
}

void init_log(int rank)
{
#ifdef USE_BOOST
    std::string fmt = boost::str(boost::format("%d: %%Message%%") % rank);

    // Output message to console
    boost::log::add_console_log(std::cout, boost::log::keywords::format = fmt, boost::log::keywords::auto_flush = true);

    boost::log::add_common_attributes();
#endif
}

DDStore::DDStore()
    : DDStore(MPI_COMM_SELF, 0, 0, 0)
{
}

DDStore::DDStore(MPI_Comm comm)
    : DDStore(comm, 0, 0, 0)
{
}

DDStore::DDStore(MPI_Comm comm, int use_mq, int role)
    : DDStore(comm, use_mq, role, 0)
{
}

DDStore::DDStore(MPI_Comm comm, int use_mq, int role, int mode)
{
    this->comm = comm;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
    init_log(this->rank);

    this->use_mq = use_mq;
    this->role = role;
    this->mode = mode;

    char *value = std::getenv("DDSTORE_VERBOSE");
    int verbose = 0;
    if (value != NULL)
    {
        verbose = atoi(value);
    }
    this->verbose = verbose;

    pthread_spin_init(&this->spinlock, 0);
    this->mutex = PTHREAD_MUTEX_INITIALIZER;

    this->ndchannel = 8;
    this->imax = 0;
}

DDStore::~DDStore()
{
    if (this->use_mq)
    {
        for (auto &x : this->qlist)
        {
            if (mq_close(x.second.mqr))
            {
                perror("mqr: mq_close");
            }
            for (int i = 0; i < this->ndchannel; i++)
            {
                if (mq_close(x.second.mqd[i]))
                {
                    perror("mqd: mq_close");
                }
            }
            if (this->role == 0)
            {
                for (int i = 0; i < this->ndchannel; i++)
                {
                    // producer destroys the queue
                    if (mq_unlink(x.second.mqd_name[i].c_str()))
                    {
                        perror("mqd: mq_unlink");
                    }
                }
            }
            if (this->role == 1)
            {
                // consumer destroys the queue
                if (mq_unlink(x.second.mqr_name.c_str()))
                {
                    perror("mqr: mq_unlink");
                }
            }
        }
    }
    this->free();
}

void DDStore::query(std::string name, VarInfo_t &varinfo)
{
    varinfo = this->varlist[name];
}

void DDStore::epoch_begin()
{
    if (!this->use_mq || (this->use_mq && (this->role == 0)))
    {
        for (auto &x : this->varlist)
        {
            if (x.second.fence_active)
                throw std::logic_error("Fence already activated");
            MPI_Win_fence(0, x.second.win);
            x.second.fence_active = true;
        }
    }
}

void DDStore::epoch_end()
{
    if (!this->use_mq || (this->use_mq && (this->role == 0)))
    {
        for (auto &x : this->varlist)
        {
            if (not x.second.fence_active)
                throw std::logic_error("Fence is not activated");
            MPI_Win_fence(0, x.second.win);
            x.second.fence_active = false;
        }
    }
}

void DDStore::free()
{
    int flag;
    MPI_Finalized(&flag);
    if (!flag)
    {
        for (auto &x : this->varlist)
        {
            if (x.second.active)
            {
                if (!this->use_mq || (this->use_mq && (this->role == 0)))
                {
                    MPI_Win_free(&x.second.win);
                    MPI_Free_mem(x.second.base);
                }
            }
            x.second.active = false;
        }
    }
}

// role: producer (0) or consumer (1)
void DDStore::queue_init(std::string name)
{
    mqd_t mqd[64];
    mqd_t mqr;
    char *mqd_name_list[64]; // MAX=64
    char mqr_name[128];
    long mqd_msgsize = 0;
    long mqr_msgsize = sizeof(Request_t);

    // VarInfo_t &varinfo = this->varlist[name];
    // std::vector<int> &lenlist = varinfo.lenlist;
    // for (long unsigned int i = 0; i < lenlist.size(); i++)
    //     if (lenlist[i] > mqd_msgsize)
    //         mqd_msgsize = lenlist[i];
    // mqd_msgsize *=  varinfo.itemsize * varinfo.disp;
    mqd_msgsize = (long)Q_ATTR_MSG_SIZE;

    for (int i = 0; i < this->ndchannel; i++) 
    {
        mqd_name_list[i] = (char*)malloc(128);
        snprintf(mqd_name_list[i], 128, "%sd-%s-%d-%d", Q_NAME, name.c_str(), this->rank, i);
    }
    snprintf(mqr_name, 128, "%sr-%s-%d", Q_NAME, name.c_str(), this->rank);

    if (this->role == 0) // producer
    {
        // producer
        struct mq_attr q_attr = {
            .mq_flags = Q_ATTR_FLAGS,     /* Flags: 0 or O_NONBLOCK */
            .mq_maxmsg = Q_ATTR_MAX_MSG,  /* Max. # of messages on queue */
            .mq_msgsize = mqd_msgsize,    /* Max. message size (bytes) */
            .mq_curmsgs = Q_ATTR_CURMSGS, /* # of messages currently in queue */
        };
        printf("[%d:%d] mqd_name0: %s\n", this->role, this->rank, mqd_name_list[0]);
        printf("[%d:%d] mqd_msgsize: %ld\n", this->role, this->rank, mqd_msgsize);
        // setup data mq
        for (int i = 0; i < this->ndchannel; i++) 
        {
            if ((mqd[i] = mq_open(mqd_name_list[i], Q_OFLAGS_PRODUCER, Q_MODE, &q_attr)) == (mqd_t)-1)
            {
                perror("producer: mqd open");
                return;
            }
        }

        // setup req mq
        while ((mqr = mq_open(mqr_name, Q_OFLAGS_CONSUMER)) == (mqd_t)-1)
        {
            if (errno == ENOENT)
            {
                printf("[%d:%d] producer: Waiting for consumer to create message queue...\n", this->role, this->rank);
                usleep(Q_CREATE_WAIT_US);
                continue;
            }
            perror("producer: mqr open");
            return;
        }
    }
    else
    {
        // consumer
        struct mq_attr q_attr = {
            .mq_flags = Q_ATTR_FLAGS,     /* Flags: 0 or O_NONBLOCK */
            .mq_maxmsg = Q_ATTR_MAX_MSG,  /* Max. # of messages on queue */
            .mq_msgsize = mqr_msgsize,    /* Max. message size (bytes) */
            .mq_curmsgs = Q_ATTR_CURMSGS, /* # of messages currently in queue */
        };
        printf("[%d:%d] mqr_name: %s\n", this->role, this->rank, mqr_name);
        printf("[%d:%d] mqr_msgsize: %ld\n", this->role, this->rank, mqr_msgsize);
        // setup req mq
        if ((mqr = mq_open(mqr_name, Q_OFLAGS_PRODUCER, Q_MODE, &q_attr)) == (mqd_t)-1)
        {
            perror("consumer: mqr open");
            return;
        }

        // setup data mq
        for (int i = 0; i < this->ndchannel; i++) 
        {
            while ((mqd[i] = mq_open(mqd_name_list[i], Q_OFLAGS_CONSUMER)) == (mqd_t)-1)
            {
                if (errno == ENOENT)
                {
                    printf("[%d:%d] consumer: Waiting for producer to create message queue...\n", this->role, this->rank);
                    usleep(Q_CREATE_WAIT_US);
                    continue;
                }
                perror("consumer: mqd open");
                return;
            }
        }
    }

    QueInfo_t queinfo;
    for (int i = 0; i < this->ndchannel; i++) \
    {
        queinfo.mqd[i] = mqd[i];
        queinfo.mqd_name[i] = std::string(mqd_name_list[i]);
    }
    queinfo.mqr = mqr;
    queinfo.mqr_name = std::string(mqr_name);
    queinfo.mqd_msgsize = mqd_msgsize;
    queinfo.mqr_msgsize = mqr_msgsize;

    this->qlist.insert(std::pair<std::string, QueInfo_t>(name, queinfo));
}

void DDStore::pushr(mqd_t mq, char *buffer, long size)
{
    if (this->verbose)
        printf("[%d:%d] pushr: %ld\n", this->role, this->rank, size);

    struct mq_attr attr;
    mq_getattr(mq, &attr);
    if (size > attr.mq_msgsize)
    {
        perror("pushr: too big");
        return;
    }

    int rc;
    rc = mq_send(mq, buffer, size, 0);
    if (rc < 0)
    {
        perror("pushr: send error");
    }
    if (this->verbose)
        printf("[%d:%d] pushr: send (%d)\n", this->role, this->rank, rc);
}

void DDStore::pullr(mqd_t mq, char *buffer, long size)
{
    int rc;
    struct mq_attr attr;

    mq_getattr(mq, &attr);
    // printf("mq_flags %ld\n", attr.mq_flags);
    // printf("mq_maxmsg %ld\n", attr.mq_maxmsg);
    // printf("mqd_msgsize %ld\n", attr.mq_msgsize);
    // printf("mq_curmsgs %ld\n", attr.mq_curmsgs);
    if (size > attr.mq_msgsize)
    {
        perror("pullr: too big");
        return;
    }

    memset(buffer, 0, size);
    if (this->verbose)
        printf("[%d:%d] pullr: ready to receive\n", this->role, this->rank);
    rc = mq_receive(mq, buffer, size, NULL);
    if (rc < 0)
    {
        perror("pullr: recv error");
        usleep(1000);
    }
    // timeout version
    // rc = -1;
    // while (rc < 0)
    // {
    //     struct timespec tm;
    //     clock_gettime(CLOCK_REALTIME, &tm);
    //     tm.tv_sec += 1;  // Set for 1 seconds
    //     rc = mq_timedreceive(mq, buffer, size, NULL, &tm);
    //     // printf("[%d:%d] pullr: done with receive\n", this->role, this->rank);
    //     if (rc < 0)
    //     {
    //         perror("pullr: recv error");
    //         sleep(1);
    //     }
    //     else
    //     {
    //         printf("[%d:%d] pullr: recv (%d)\n", this->role, this->rank, rc);
    //     }
    // }
    // printf("[%d:%d] pullr: done with receive\n", this->role, this->rank);
}

void DDStore::pushd(mqd_t mq, char *buffer, long size)
{
    // 1. calculate nchunk
    // 2. send nchunk info first
    // 3. send data in chunk
    struct mq_attr attr;
    mq_getattr(mq, &attr);
    // printf("mq_flags %ld\n", attr.mq_flags);
    // printf("mq_maxmsg %ld\n", attr.mq_maxmsg);
    // printf("mqd_msgsize %ld\n", attr.mq_msgsize);
    // printf("mq_curmsgs %ld\n", attr.mq_curmsgs);

    int nchunk = size / attr.mq_msgsize;
    if (size > nchunk * attr.mq_msgsize)
        nchunk += 1;
    if (this->verbose)
        printf("[%d:%d] pushd: %ld %d\n", this->role, this->rank, size, nchunk);

    int rc;
    rc = mq_send(mq, (const char *)&nchunk, sizeof(int), 0);
    if (rc < 0)
    {
        perror("pushd: send head error");
    }
    if (this->verbose)
        printf("[%d:%d] pushd: send head (%d)\n", this->role, this->rank, rc);

    int nbytes = 0;
    for (int i = 0; i < nchunk; i++)
    {
        int len = attr.mq_msgsize;
        if (i == nchunk - 1)
            len = size - i * attr.mq_msgsize;

        rc = mq_send(mq, buffer + i * attr.mq_msgsize, len, 0);
        if (rc < 0)
        {
            perror("pushd: send data error");
            i--;
            continue;
        }
        nbytes += len;
        if (this->verbose)
            printf("[%d:%d] pushd: send data (%d), i,total: %d %d\n", this->role, this->rank, rc, i, nbytes);
    }
}

void DDStore::pulld(mqd_t mq, char *buffer, long size)
{
    int rc;
    int nchunk = 0;
    struct mq_attr attr;
    mq_getattr(mq, &attr);
    // char msg[Q_ATTR_MSG_SIZE];

    rc = mq_receive(mq, (char *)&nchunk, attr.mq_msgsize, NULL);
    if (rc < 0)
    {
        perror("pulld: recv head error");
    }
    if (this->verbose)
        printf("[%d:%d] pulld: %ld %d\n", this->role, this->rank, size, nchunk);

    memset(buffer, 0, size);

    int nbytes = 0;
    for (int i = 0; i < nchunk; i++)
    {
        rc = mq_receive(mq, buffer + i * attr.mq_msgsize, attr.mq_msgsize, NULL);
        if (rc < 0)
        {
            perror("pulld: recv data error");
            i--;
            continue;
        }
        nbytes += rc;
        if (this->verbose)
            printf("[%d:%d] pulld: recv data (%d), i,total: %d %d\n", this->role, this->rank, rc, i, nbytes);
    }
}
