#include "ddstore.hpp"
#include <stdio.h>
#include <stdlib.h>

int sortedsearch(std::vector<long> &vec, long num)
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

DDStore::DDStore()
{
    this->comm = MPI_COMM_SELF;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
}

DDStore::DDStore(MPI_Comm comm)
{
    this->comm = comm;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
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
            {
                MPI_Win_free(&x.second.win);
                MPI_Free_mem(x.second.base);
            }
            x.second.active = false;
        }
    }
}
