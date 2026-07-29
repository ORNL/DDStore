#include "ddstore.hpp"
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <algorithm>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

int sortedsearch(const std::vector<long> &vec, long num)
{
    if (vec.empty() || num < 0 || num >= vec.back())
        throw std::out_of_range(
            "Global index " + std::to_string(num) +
            " is out of range [0, " + std::to_string(vec.empty() ? 0 : vec.back()) + ")");

    return (int)std::distance(vec.begin(),
                              std::upper_bound(vec.begin(), vec.end(), num));
}

DDStore::DDStore() : method(0)
{
    this->comm = MPI_COMM_SELF;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
}

DDStore::DDStore(MPI_Comm comm) : method(0)
{
    this->comm = comm;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
}

DDStore::DDStore(int method, MPI_Comm comm)
{
    this->method = method;
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
    varinfo = this->varlist.at(name);
}

void DDStore::epoch_begin()
{
    if (!this->method)
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
    if (!this->method)
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
    if (!this->method && !flag)
    {
        for (auto &x : this->varlist)
        {
            if (x.second.active)
            {
                MPI_Win_free(&x.second.win);
                // (2024/12) no need as using the user pointer
                // MPI_Free_mem(x.second.base);
            }
            x.second.active = false;
        }
    }
    else if (this->method == 1)
    {
        for (auto &x : this->varlist)
        {
            if (x.second.active && x.second.fabric_state)
            {
                struct fabric_state *fs = x.second.fabric_state;
                if (fs->recv_mr)       fi_close(&fs->recv_mr->fid);
                if (fs->mr)            fi_close(&fs->mr->fid);
                if (fs->signal)        fi_close(&fs->signal->fid);
                if (fs->cq_signal)     fi_close(&fs->cq_signal->fid);
                if (fs->av)            fi_close(&fs->av->fid);
                if (fs->domain)        fi_close(&fs->domain->fid);
                if (fs->fabric)        fi_close(&fs->fabric->fid);
                if (fs->info)          fi_freeinfo(fs->info);
                if (fs->ctx)           ::free(fs->ctx);
                ::free(fs->comm_partner);
                ::free(fs->remote_key);
                ::free(fs->remote_address);
                ::free(fs);
                x.second.fabric_state = NULL;
            }
            x.second.active = false;
        }
    }
}
