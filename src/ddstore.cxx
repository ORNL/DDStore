#include "ddstore.hpp"
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <algorithm>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void check_method(int method)
{
    if (method < 0 || method > 2)
        throw std::invalid_argument("method must be 0, 1 or 2 (got " + std::to_string(method) + ")");
}

#ifdef DDSTORE_HAVE_MPI
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
    check_method(method);
    this->method = method;
    this->comm = comm;
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
    if (method != 0)
        this->exchange.reset(new MpiExchange(comm, this->rank, this->comm_size));
}

/* Method 2: core member constructor. n_core is derived from comm_size —
 * every rank in `comm` is assumed to be a core member.                      */
DDStore::DDStore(int method, MPI_Comm comm,
                 const std::string &handshake_dir)
    : is_extra(false)
{
    check_method(method);
    this->method        = method;
    this->comm          = comm;
    this->handshake_dir = resolve_dir(handshake_dir);
    MPI_Comm_size(this->comm, &this->comm_size);
    MPI_Comm_rank(this->comm, &this->rank);
    this->n_core        = this->comm_size;
    if (method != 0)
        this->exchange.reset(new MpiExchange(comm, this->rank, this->comm_size));
    fprintf(stderr, "[DDStore] method=2 core: handshake_dir=%s, n_core=%d\n",
            this->handshake_dir.c_str(), this->n_core);
}
#endif

/* Methods 1/2 without MPI: records are exchanged through files in dir.     */
DDStore::DDStore(int method, const std::string &dir, int rank, int size,
                 const std::string &ns)
    : DDStore(method,
              new FileExchange(resolve_dir(dir), rank, size,
                               ns.empty() ? next_exchange_ns() : ns),
              dir)
{
}

DDStore::DDStore(int method, Exchange *x, const std::string &dir)
    : comm_size(x->size), rank(x->rank), n_core(0), is_extra(false)
{
    this->exchange.reset(x);
    if (method != 1 && method != 2)
        throw std::invalid_argument(
            "a store without an MPI communicator needs method 1 or 2 (got " +
            std::to_string(method) + ")");
    this->method = method;
    if (method == 2 || !dir.empty())
        this->handshake_dir = resolve_dir(dir);
    if (method == 2)
    {
        this->n_core = this->comm_size;
        fprintf(stderr, "[DDStore] method=2 core (no MPI): handshake_dir=%s, rank=%d, n_core=%d\n",
                this->handshake_dir.c_str(), this->rank, this->comm_size);
    }
}

/* Method 2: extra member constructor (no MPI communicator needed). */
DDStore::DDStore(int method, const std::string &handshake_dir, int n_core)
    : comm_size(1), rank(0), is_extra(true)
{
    if (method != 2)
        throw std::invalid_argument("an extra member needs method 2 (got " +
                                    std::to_string(method) + ")");
    this->method        = method;
    this->handshake_dir = resolve_dir(handshake_dir);
    this->n_core        = n_core;
    fprintf(stderr, "[DDStore] method=2 extra: handshake_dir=%s\n",
            this->handshake_dir.c_str());
}

DDStore *DDStore::from_fortran_comm(int method, long long fcomm,
                                    const std::string &handshake_dir)
{
#ifdef DDSTORE_HAVE_MPI
    MPI_Comm comm = MPI_Comm_f2c((MPI_Fint)fcomm);
    if (method == 2)
        return new DDStore(method, comm, handshake_dir);
    return new DDStore(method, comm);
#else
    (void)method;
    (void)fcomm;
    (void)handshake_dir;
    throw std::runtime_error(
        "DDStore was built without MPI (DDSTORE_MPI=0): pass no communicator "
        "and use method 1 or 2 with rank/size");
#endif
}

DDStore::~DDStore()
{
    this->free();
}

void DDStore::query(std::string name, VarInfo_t &varinfo)
{
    varinfo = this->varlist.at(name);
}

/* Method 0: synchronization points, not RMA epochs. get() locks its target
 * itself (MPI_Win_lock/unlock), which MPI doesn't allow inside a fence epoch
 * (Open MPI then reads nothing). epoch_begin() makes this rank's writes
 * (update()) visible to the others' reads (MPI_Win_sync, in a passive-target
 * epoch of its own), then waits for every rank; epoch_end() waits until
 * every rank is done reading, so no rank updates while others read.       */
void DDStore::epoch_begin()
{
#ifdef DDSTORE_HAVE_MPI
    if (!this->method && !this->varlist.empty())
    {
        for (auto &x : this->varlist)
            if (x.second.active && x.second.epoch_active)
                throw std::logic_error("Epoch already begun");
        for (auto &x : this->varlist)
        {
            if (!x.second.active)
                continue;
            ddstore_mpi_check(MPI_Win_lock_all(MPI_MODE_NOCHECK, x.second.win), "MPI_Win_lock_all");
            ddstore_mpi_check(MPI_Win_sync(x.second.win), "MPI_Win_sync");
            ddstore_mpi_check(MPI_Win_unlock_all(x.second.win), "MPI_Win_unlock_all");
            x.second.epoch_active = true;
        }
        MPI_Barrier(this->comm);
    }
#endif
    /* Methods 1 and 2 use libfabric: no epoch needed. */
}

void DDStore::epoch_end()
{
#ifdef DDSTORE_HAVE_MPI
    if (!this->method && !this->varlist.empty())
    {
        for (auto &x : this->varlist)
            if (x.second.active && !x.second.epoch_active)
                throw std::logic_error("Epoch not begun");
        for (auto &x : this->varlist)
            x.second.epoch_active = false;
        MPI_Barrier(this->comm);
    }
#endif
    /* Methods 1 and 2 use libfabric: no epoch needed. */
}

/* --------------------------------------------------------------------------
 * join() — extra member: discover a variable published by core members.
 *
 * Calls handshake_join(), which polls for the combined {name}.bin record
 * file written by core rank 0, then populates a fabric_state and builds
 * the lenlist for get() calls.
 * -------------------------------------------------------------------------- */
void DDStore::join(std::string name)
{
    if (!this->is_extra)
        throw std::logic_error("join() is only valid for extra members");
    if (this->method != 2)
        throw std::logic_error("join() requires method=2");
    this->new_name(name);

    struct fabric_state *fs =
        (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
    pthread_mutex_init(&fs->recv_lock, NULL);
    fs->world_size = this->n_core;
    fs->rank       = -1; /* extra members have no core rank */

    init_fabric(fs);
    if (!fs->info)
        throw std::runtime_error("init_fabric failed for extra member");

    /* Extra member has no send buffer, so nothing is registered for remote
     * access (mr stays NULL, key 0). It only issues fi_read()s; get()
     * registers each destination buffer as usual in read_from_remote().     */
    fs->send_data     = NULL;
    fs->send_data_len = 0;
    fs->mr            = NULL;
    fs->key           = 0;

    std::vector<long> raw_lens(this->n_core);
    std::vector<struct CoreRecord> records(this->n_core);
    int out_disp = 0, out_itemsize = 0;
    if (handshake_join(fs,
                       this->handshake_dir.c_str(), name.c_str(),
                       this->n_core,
                       raw_lens.data(), &out_disp, &out_itemsize,
                       records.data()) != 0)
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
    var.lenlist      = lenlist;
    var.active       = true;
    var.epoch_active = false;
    var.base         = NULL; /* extra member owns no data */
    var.owns_base    = false;
    var.fabric_state = fs;
    var.records      = records;
    this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
}

DDStore::DDStore(int method, int n_core, reader_tag)
    : method(method), comm_size(1), rank(0), n_core(n_core), is_extra(true)
{
}

/* --------------------------------------------------------------------------
 * fork_reader() — read-only store for a forked process (see ddstore.hpp).
 *
 * Per variable: a new fabric_state with its own fabric/domain/endpoint/AV
 * (init_fabric), the records inserted into that AV (use_records), and no
 * send MR. get() and get_batch() then work as for an extra member.
 * -------------------------------------------------------------------------- */
DDStore *DDStore::fork_reader() const
{
    if (this->method != 1 && this->method != 2)
        throw std::runtime_error(
            "a forked process can only read a method 1 or 2 store "
            "(method 0 is MPI, which a forked process can't use)");
    const char *mon = getenv("FI_MR_CACHE_MONITOR");
    if (getpid() != this->owner_pid && mon && strcmp(mon, "kdreg2") == 0)
        throw std::runtime_error(
            "DDStore read from a forked process (DataLoader worker) needs "
            "DDSTORE_FORK_WORKERS=1 (or FI_MR_CACHE_MONITOR=memhooks) set "
            "before the store is created: the kdreg2 MR cache monitor "
            "doesn't survive fork and the worker's reads would hang");

    std::unique_ptr<DDStore> r(new DDStore(this->method, this->n_core, reader_tag()));
    for (const auto &x : this->varlist)
    {
        const VarInfo_t &src = x.second;
        if (!src.active)
            continue;
        if (src.records.empty())
            throw std::logic_error("no records for variable " + src.name);

        struct fabric_state *fs =
            (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
        if (!fs)
            throw std::bad_alloc();
        pthread_mutex_init(&fs->recv_lock, NULL);
        fs->rank = -1;

        VarInfo_t var;
        var.name         = src.name;
        var.itemsize     = src.itemsize;
        var.disp         = src.disp;
        var.lenlist      = src.lenlist;
        var.active       = true;
        var.epoch_active = false;
        var.base         = NULL;
        var.owns_base    = false;
        var.fabric_state = fs;
        var.records      = src.records;
        /* In the map before anything can throw, so r's free() cleans up. */
        r->varlist.insert(std::pair<std::string, VarInfo_t>(var.name, var));

        init_fabric(fs);
        if (!fs->info)
            throw std::runtime_error("init_fabric failed in forked reader (variable " +
                                     src.name + ")");
        use_records(fs, src.records.data(), (int)src.records.size());
    }
    return r.release();
}

/* --------------------------------------------------------------------------
 * free() — release all resources.
 *
 * Per variable: the MPI window (method 0) or the libfabric objects (methods
 * 1/2) first, since they reference the buffer, then the buffer itself if
 * DDStore allocated it (owns_base). MPI_Win_free/MPI_Free_mem are skipped
 * after MPI_Finalize (no longer callable). Idempotent via `active`.
 * -------------------------------------------------------------------------- */
void DDStore::free()
{
    if (getpid() != this->owner_pid)
    {
        /* Inherited through fork: the parent still owns these resources. */
        for (auto &x : this->varlist)
        {
            x.second.fabric_state = NULL;
            x.second.base = NULL;
            x.second.owns_base = false;
            x.second.active = false;
        }
        return;
    }

    int finalized = 0;
#ifdef DDSTORE_HAVE_MPI
    MPI_Finalized(&finalized);
#endif
    (void)finalized;
    for (auto &x : this->varlist)
    {
        VarInfo_t &var = x.second;
        if (!var.active)
            continue;

        if (this->method == 0)
        {
#ifdef DDSTORE_HAVE_MPI
            if (!finalized)
                MPI_Win_free(&var.win);
#endif
        }
        else if (var.fabric_state)
        {
            struct fabric_state *fs = var.fabric_state;
            close_recv_regions(fs);
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
            pthread_mutex_destroy(&fs->recv_lock);
            ::free(fs);
            var.fabric_state = NULL;
        }

        if (var.owns_base && var.base)
            free_base(var.base, var.base_alloc, var.base_len);
        var.base = NULL;
        var.owns_base = false;
        var.active = false;
    }
#ifdef DDSTORE_HAVE_MPI
    if (this->coll_comm != MPI_COMM_NULL && !finalized)
        MPI_Comm_free(&this->coll_comm);
#endif
}
