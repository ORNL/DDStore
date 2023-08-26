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
{
    this->comm = MPI_COMM_SELF;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
    init_log(this->rank);
}

DDStore::DDStore(MPI_Comm comm)
{
    this->comm = comm;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
    init_log(this->rank);
}

DDStore::~DDStore()
{
    for (auto &x : this->qlist)
    {
        if (mq_close(x.second.mqr))
        {
            perror("mqr: mq_close");
        }
        if (mq_close(x.second.mqd))
        {
            perror("mqd: mq_close");
        }
        if (x.second.role == 0)
        {
            // consumer destroys the queue
            if (mq_unlink(x.second.mqr_name.c_str()))
            {
                perror("mqr: mq_unlink");
            }
        }
        if (x.second.role == 1)
        {
            // consumer destroys the queue
            if (mq_unlink(x.second.mqd_name.c_str()))
            {
                perror("mqd: mq_unlink");
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
    for (auto &x : this->varlist)
    {
        if (x.second.fence_active)
            throw std::logic_error("Fence already activated");
        MPI_Win_fence(0, x.second.win);
        x.second.fence_active = true;
    }
}

void DDStore::epoch_end()
{
    for (auto &x : this->varlist)
    {
        if (not x.second.fence_active)
            throw std::logic_error("Fence is not activated");
        MPI_Win_fence(0, x.second.win);
        x.second.fence_active = false;
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
                MPI_Win_free(&x.second.win);
            x.second.active = false;
        }
    }
}

// role: producer (0) or consumer (1)
void DDStore::queue_init(std::string name, int role)
{
    mqd_t mqd;
    mqd_t mqr;
    char mqd_name[128];
    char mqr_name[128];
    long mqd_msgsize = 0;
    long mqr_msgsize = sizeof(long unsigned int);

    VarInfo_t &varinfo = this->varlist[name];
    std::vector<int> &lenlist = varinfo.lenlist;
    for (long unsigned int i = 0; i < lenlist.size(); i++)
        if (lenlist[i] > mqd_msgsize)
            mqd_msgsize = lenlist[i];
    mqd_msgsize *=  varinfo.itemsize * varinfo.disp;


    snprintf(mqd_name, 128, "%sd-%s-%d", Q_NAME, name.c_str(), this->rank);
    snprintf(mqr_name, 128, "%sr-%s-%d", Q_NAME, name.c_str(), this->rank);

    if (role == 0)
    {
        // producer
        struct mq_attr q_attr = {
            .mq_flags = Q_ATTR_FLAGS,       /* Flags: 0 or O_NONBLOCK */
            .mq_maxmsg = Q_ATTR_MAX_MSG,    /* Max. # of messages on queue */
            .mq_msgsize = mqd_msgsize,  /* Max. message size (bytes) */
            .mq_curmsgs = Q_ATTR_CURMSGS,   /* # of messages currently in queue */
        };
        printf("mqd_name: %s\n", mqd_name);
        printf("mqd_msgsize: %ld\n", mqd_msgsize);
        // setup data mq
        if ((mqd = mq_open(mqd_name, Q_OFLAGS_PRODUCER, Q_MODE, &q_attr)) == (mqd_t)-1)
        {
            perror("produce: mq_open");
            return;
        }

        // setup req mq
        while ((mqr = mq_open(mqr_name, Q_OFLAGS_CONSUMER)) == (mqd_t)-1)
        {
            if (errno == ENOENT)
            {
                printf("produce: Waiting for consumer to create message queue...\n");
                usleep(Q_CREATE_WAIT_US);
                continue;
            }
            perror("produce: mq_open");
            return;
        }

    }
    else
    {
        // consumer
        struct mq_attr q_attr = {
            .mq_flags = Q_ATTR_FLAGS,       /* Flags: 0 or O_NONBLOCK */
            .mq_maxmsg = Q_ATTR_MAX_MSG,    /* Max. # of messages on queue */
            .mq_msgsize = mqr_msgsize,  /* Max. message size (bytes) */
            .mq_curmsgs = Q_ATTR_CURMSGS,   /* # of messages currently in queue */
        };
        printf("mqr_name: %s\n", mqr_name);
        printf("mqr_msgsize: %ld\n", mqr_msgsize);
        // setup req mq
        if ((mqr = mq_open(mqr_name, Q_OFLAGS_PRODUCER, Q_MODE, &q_attr)) == (mqd_t)-1)
        {
            perror("consume: mq_open");
            return;
        }

        // setup data mq
        while ((mqd = mq_open(mqd_name, Q_OFLAGS_CONSUMER)) == (mqd_t)-1)
        {
            if (errno == ENOENT)
            {
                printf("consume: Waiting for producer to create message queue...\n");
                usleep(Q_CREATE_WAIT_US);
                continue;
            }
            perror("consume: mq_open");
            return;
        }
    }

    QueInfo_t queinfo;
    queinfo.mqd = mqd;
    queinfo.mqr = mqr;
    queinfo.mqd_name = std::string(mqd_name);
    queinfo.mqr_name = std::string(mqr_name);
    queinfo.mqd_msgsize = mqd_msgsize;
    queinfo.mqr_msgsize = mqr_msgsize;
    queinfo.role = role;

    this->qlist.insert(std::pair<std::string, QueInfo_t>(name, queinfo));
}

void DDStore::push(mqd_t mq, char *buffer, int size)
{
    printf ("mq_send: %d\n", size);
    int rc;
    rc = mq_send(mq, buffer, size, 0);
    if (rc < 0)
    {
        perror("produce: mq_send");
    }
    else
    {
        printf("produce: send (%d)\n", rc);
    }
}

void DDStore::pull(mqd_t mq, char *buffer, int size)
{
    int rc;
    struct mq_attr attr;

    mq_getattr(mq, &attr);
    // printf("mq_flags %ld\n", attr.mq_flags);
    // printf("mq_maxmsg %ld\n", attr.mq_maxmsg);
    // printf("mqd_msgsize %ld\n", attr.mq_msgsize);
    // printf("mq_curmsgs %ld\n", attr.mq_curmsgs);
    if (attr.mq_msgsize > size)
    {
        perror("pull: too big");
        return;
    }

    memset(buffer, 0, size);
    rc = mq_receive(mq, buffer, attr.mq_msgsize, NULL);
    if (rc < 0)
    {
        perror("consume: mq_receive");
    }
    else
    {
        printf("consume: recv (%d)\n", rc);
    }
}
