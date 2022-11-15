#include "ddstore.hpp"
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_BOOST
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup.hpp>

#define LOG BOOST_LOG_TRIVIAL(debug)
#endif

int sortedsearch(std::vector<int> &vec, int num)
{
    int rtn = 0;
    for (long unsigned int i = 1; i < vec.size(); i++)
    {
        if ((vec[i - 1] <= num) && (num < vec[i]))
        {
            rtn = i;
            break;
        }
    }
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
        MPI_Win_fence(0, x.second.win);
    }
}

void DDStore::epoch_end()
{
    for (auto &x : this->varlist)
    {
        MPI_Win_fence(0, x.second.win);
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
