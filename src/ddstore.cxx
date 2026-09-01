#include "ddstore.hpp"
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <algorithm>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

/* Forward-declare the C helper from common.cxx so we can call it here.     */
extern "C" const char *resolve_handshake_dir(const char *user_dir);

/* Convenience wrapper: resolve and return as std::string.                   */
static std::string resolve_dir(const std::string &user_dir)
{
    return std::string(resolve_handshake_dir(user_dir.c_str()));
}

int sortedsearch(const std::vector<long> &vec, long num)
{
    if (vec.empty() || num < 0 || num >= vec.back())
        throw std::out_of_range(
            "Global index " + std::to_string(num) +
            " is out of range [0, " + std::to_string(vec.empty() ? 0 : vec.back()) + ")");

    return (int)std::distance(vec.begin(),
                              std::upper_bound(vec.begin(), vec.end(), num));
}

DDStore::DDStore() : method(0), comm_size(1), rank(0), n_core(0), is_extra(false)
{
    this->comm = MPI_COMM_SELF;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
}

DDStore::DDStore(MPI_Comm comm) : method(0), n_core(0), is_extra(false)
{
    this->comm = comm;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
}

DDStore::DDStore(int method, MPI_Comm comm) : n_core(0), is_extra(false)
{
    this->method = method;
    this->comm = comm;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
}

/* Method 2: core member constructor. n_core is derived from comm_size —
 * every rank in `comm` is assumed to be a core member.                      */
DDStore::DDStore(int method, MPI_Comm comm,
                 const std::string &handshake_dir)
    : is_extra(false)
{
    this->method        = method;
    this->comm          = comm;
    this->handshake_dir = resolve_dir(handshake_dir);
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
    this->n_core        = this->comm_size;
    fprintf(stderr, "[DDStore] method=2 core: handshake_dir=%s, n_core=%d\n",
            this->handshake_dir.c_str(), this->n_core);
}

/* Method 2: extra member constructor (no MPI communicator needed). */
DDStore::DDStore(int method, const std::string &handshake_dir, int n_core)
    : comm(MPI_COMM_SELF), comm_size(1), rank(0), is_extra(true)
{
    this->method        = method;
    this->handshake_dir = resolve_dir(handshake_dir);
    this->n_core        = n_core;
    fprintf(stderr, "[DDStore] method=2 extra: handshake_dir=%s\n",
            this->handshake_dir.c_str());
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
    /* Methods 1 and 2 use libfabric — no fence needed. */
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
    /* Methods 1 and 2 use libfabric — no fence needed. */
}

/* --------------------------------------------------------------------------
 * join() — extra member: discover a variable published by core members.
 *
 * Calls handshake_join() which polls for all CoreRecord files, then
 * populates a fabric_state and builds the lenlist for get() calls.
 * -------------------------------------------------------------------------- */
void DDStore::join(std::string name)
{
    if (!this->is_extra)
        throw std::logic_error("join() is only valid for extra members");
    if (this->method != 2)
        throw std::logic_error("join() requires method=2");

    struct fabric_state *fs =
        (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
    fs->world_size = this->n_core;
    fs->rank       = -1; /* extra members have no core rank */

    init_fabric(fs);
    if (!fs->info)
        throw std::runtime_error("init_fabric failed for extra member");

    /* Extra member has no send buffer to register as MR — set a dummy
     * zero-length registration so handshake_join doesn't need special-casing.
     * We only need fi_read capability, not FI_REMOTE_READ on our side.      */
    fs->send_data     = NULL;
    fs->send_data_len = 0;
    fs->mr            = NULL;
    fs->key           = 0;

    std::vector<long> raw_lens(this->n_core);
    int out_disp = 0, out_itemsize = 0;
    if (handshake_join(fs,
                       this->handshake_dir.c_str(), name.c_str(),
                       this->n_core,
                       raw_lens.data(), &out_disp, &out_itemsize) != 0)
        throw std::runtime_error("handshake_join failed for variable: " + name);

    /* Build prefix-sum lenlist. */
    long sum = 0;
    std::vector<long> lenlist(this->n_core);
    for (int i = 0; i < this->n_core; i++)
    {
        sum += raw_lens[i];
        lenlist[i] = sum;
    }

    VarInfo_t var;
    var.name         = name;
    var.itemsize     = out_itemsize;
    var.disp         = out_disp;
    var.win          = MPI_WIN_NULL;
    var.lenlist      = lenlist;
    var.active       = true;
    var.fence_active = false;
    var.base         = NULL; /* extra member owns no data */
    var.fabric_state = fs;
    this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
}

/* --------------------------------------------------------------------------
 * free() — release all resources.
 * -------------------------------------------------------------------------- */
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
            }
            x.second.active = false;
        }
    }
    else if (this->method == 1 || this->method == 2)
    {
        for (auto &x : this->varlist)
        {
            if (x.second.active && x.second.fabric_state)
            {
                struct fabric_state *fs = x.second.fabric_state;
                if (fs->recv_mr)   fi_close(&fs->recv_mr->fid);
                if (fs->mr)        fi_close(&fs->mr->fid);
                if (fs->signal)    fi_close(&fs->signal->fid);
                if (fs->cq_signal) fi_close(&fs->cq_signal->fid);
                if (fs->av)        fi_close(&fs->av->fid);
                if (fs->domain)    fi_close(&fs->domain->fid);
                if (fs->fabric)    fi_close(&fs->fabric->fid);
                if (fs->info)      fi_freeinfo(fs->info);
                if (fs->ctx)       ::free(fs->ctx);
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
