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
    for (auto &x : this->quelist)
    {
        if (mq_close(x.second.mq))
        {
            perror("produce: mq_close");
        }
        if (x.second.role == 1)
        {
            // consumer destroys the queue
            if (mq_unlink(x.second.mq_name.c_str()))
            {
                perror("consume: mq_unlink");
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
    mqd_t mq;
    char mqname[128];
    long mq_msgsize = 0;

    VarInfo_t &varinfo = this->varlist[name];
    std::vector<int> &lenlist = varinfo.lenlist;
    for (long unsigned int i = 0; i < lenlist.size(); i++)
        if (lenlist[i] > mq_msgsize)
            mq_msgsize = lenlist[i];
    mq_msgsize *=  varinfo.itemsize * varinfo.disp;

  struct mq_attr q_attr = {
    .mq_flags = Q_ATTR_FLAGS,       /* Flags: 0 or O_NONBLOCK */
    .mq_maxmsg = Q_ATTR_MAX_MSG,    /* Max. # of messages on queue */
    .mq_msgsize = mq_msgsize,  /* Max. message size (bytes) */
    .mq_curmsgs = Q_ATTR_CURMSGS,   /* # of messages currently in queue */
  };

    snprintf(mqname, 128, "%s-%s-%d", Q_NAME, name.c_str(), this->rank);

    if (role == 0)
    {
        // producer
        printf("mqname: %s\n", mqname);
        printf("mq_msgsize: %ld\n", mq_msgsize);
        if ((mq = mq_open(mqname, Q_OFLAGS_PRODUCER, Q_MODE, &q_attr)) == (mqd_t)-1)
        {
            perror("produce: mq_open");
            return;
        }
    }
    else
    {
        while ((mq = mq_open(mqname, Q_OFLAGS_CONSUMER)) == (mqd_t)-1)
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
    queinfo.mq = mq;
    queinfo.mq_name = std::string(mqname);
    queinfo.mq_msgsize = mq_msgsize;
    queinfo.role = role;

    this->quelist.insert(std::pair<std::string, QueInfo_t>(name, queinfo));
}

void DDStore::push(std::string name, char *buffer, int size)
{
    QueInfo_t queinfo = this->quelist[name];

    printf ("mq_send: %d\n", size);
    int rc;
    rc = mq_send(queinfo.mq, buffer, size, 0);
    if (rc < 0)
    {
        perror("produce: mq_send");
    }
    else
    {
        printf("produce: send (%d)\n", rc);
    }
}

void DDStore::pull(std::string name, char *buffer, int size)
{
    QueInfo_t queinfo = this->quelist[name];

    int rc;
    struct mq_attr attr;

    mq_getattr(queinfo.mq, &attr);
    printf("mq_flags %ld\n", attr.mq_flags);
    printf("mq_maxmsg %ld\n", attr.mq_maxmsg);
    printf("mq_msgsize %ld\n", attr.mq_msgsize);
    printf("mq_curmsgs %ld\n", attr.mq_curmsgs);
    if (attr.mq_msgsize > size)
    {
        perror("pull: too big");
        return;
    }

    memset(buffer, 0, size);
    rc = mq_receive(queinfo.mq, buffer, attr.mq_msgsize, NULL);
    if (rc < 0)
    {
        perror("consume: mq_receive");
    }
    else
    {
        printf("consume: recv (%d)\n", rc);
    }
}
