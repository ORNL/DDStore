#include <iostream>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <mpi.h>
#include <string>
#include <typeinfo>
#include <vector>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include "common.h"

struct VarInfo
{
    std::string name;
    int itemsize;
    int disp;
    std::vector<long> lenlist;
    MPI_Win win;
    bool active;
    bool fence_active;
    void *base;
    /* true if base came from MPI_Alloc_mem() in add()/init() and free()
     * must release it; false for a caller-owned GPU buffer (add() with
     * hmem_iface != 0) or a joined variable (base == NULL).               */
    bool owns_base;
    struct fabric_state *fabric_state;
};
typedef struct VarInfo VarInfo_t;

int sortedsearch(const std::vector<long> &vec, long num);

class DDStore
{
public:
    DDStore();
    DDStore(MPI_Comm comm);
    DDStore(int method, MPI_Comm comm);

    /* Method 2: core member constructor.
     * handshake_dir: shared directory visible to all processes.
     * n_core is derived from the communicator size (all ranks in `comm`
     * are assumed to be core members).                                      */
    DDStore(int method, MPI_Comm comm,
            const std::string &handshake_dir);

    /* Method 2: extra member constructor (no MPI communicator required).
     * The extra member calls join() per variable to discover the data.      */
    DDStore(int method, const std::string &handshake_dir, int n_core);

    ~DDStore();

    void query(std::string name, VarInfo_t &varinfo);

    /* Total row count across all core ranks for a variable (added or joined). */
    long size(std::string name)
    {
        const VarInfo_t& varinfo = this->varlist.at(name);
        return varinfo.lenlist.empty() ? 0 : varinfo.lenlist.back();
    }

    void epoch_begin();
    void epoch_end();
    void free();

    /* Method 2 extra member: discover variable published by core members.   */
    void join(std::string name);

    /* DDSTORE_PROFILE=1 counters for `name` (methods 1/2), as
     * {calls, lock_wait_ns, mr_ns, mr_miss, read_ns, cq_ns, rows}; all zero
     * for method 0 or when profiling is off. Takes the variable's lock.     */
    void profile(std::string name, unsigned long long out[7])
    {
        for (int i = 0; i < 7; i++)
            out[i] = 0;
        const VarInfo_t &varinfo = this->varlist.at(name);
        struct fabric_state *fs = varinfo.fabric_state;
        if (!fs)
            return;
        fabric_state_lock_guard lock(fs);
        out[0] = fs->prof_calls;
        out[1] = fs->prof_lock_wait_ns;
        out[2] = fs->prof_mr_ns;
        out[3] = fs->prof_mr_miss;
        out[4] = fs->prof_read_ns;
        out[5] = fs->prof_cq_ns;
        out[6] = fs->prof_rows;
    }

    /* hmem_iface: 0 (FI_HMEM_SYSTEM) for a host buffer, or an fi_hmem_iface
     * value (FI_HMEM_CUDA, FI_HMEM_ROCR, ...) identifying what kind of GPU
     * memory `buffer` is. Mirrors get()'s hmem_iface parameter.
     *
     * LIFETIME CONTRACT: for hmem_iface == 0 (host), DDStore makes its own
     * private copy of `buffer` (as it always has) -- the caller's buffer
     * may be freed/reused immediately after add() returns. For
     * hmem_iface != 0 (GPU), DDStore does NOT copy -- it registers the
     * caller's own device pointer directly. The caller must keep that GPU
     * allocation alive (not garbage-collected, not reused) for as long as
     * this variable stays registered, i.e. until free() or this DDStore's
     * destruction. pyddstore/_core.pyx enforces this for Python callers via a
     * keepalive dict; direct C++ callers must manage it themselves.         */
    template <typename T>
    void add(std::string name, T *buffer, long nrows, int disp, int hmem_iface = 0)
    {
        if (this->method == 0 && hmem_iface != 0)
            throw std::runtime_error("GPU source buffer is not supported with method=0 (MPI_Win)");

        void *base = NULL;
        if (hmem_iface == 0)
        {
            // (2025/03) jyc: necessary to avoid memory error
            int err = MPI_Alloc_mem((MPI_Aint)(nrows * disp * sizeof(T)), MPI_INFO_NULL, &base);
            if (err)
            {
                exit(1);
            }
            memcpy(base, buffer, nrows * disp * sizeof(T));
        }
        else
        {
            /* GPU source buffer: register the caller's own pointer
             * directly. No MPI_Alloc_mem, no copy -- see lifetime
             * contract above.                                              */
            base = (void *)buffer;
        }

        MPI_Win win = MPI_WIN_NULL;
        struct fabric_state *fabric_state = NULL;

        if (this->method == 0)
        {
            MPI_Win_create(base,                               /* pre-allocated buffer */
                           (MPI_Aint)nrows * disp * sizeof(T), /* size in bytes */
                           disp * sizeof(T),                   /* displacement units */
                           MPI_INFO_NULL,                      /* info object */
                           this->comm,                         /* communicator */
                           &win /* window object */);
        }
        else if (this->method == 1)
        {
            fabric_state = (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
            pthread_mutex_init(&fabric_state->recv_lock, NULL);
            fabric_state->send_data       = (char *)base;
            fabric_state->send_data_len   = nrows * disp * sizeof(T);
            fabric_state->send_hmem_iface = hmem_iface;
            fabric_state->world_size      = this->comm_size;
            fabric_state->rank            = this->rank;

            init_fabric(fabric_state);
            if (!fabric_state->info)
                throw std::runtime_error("init_fabric failed: no suitable fabric found");
            if (hmem_iface != 0 && !is_hmem_capable(fabric_state))
                throw std::runtime_error(
                    "GPU source buffer requires DDSTORE_FABRIC=cxi "
                    "(current fabric does not support FI_HMEM)");
            if (handshake(fabric_state, this->comm) != 0)
                throw std::runtime_error("handshake failed (method=1)");
        }
        else if (this->method == 2)
        {
            fabric_state = (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
            pthread_mutex_init(&fabric_state->recv_lock, NULL);
            fabric_state->send_data       = (char *)base;
            fabric_state->send_data_len   = nrows * disp * sizeof(T);
            fabric_state->send_hmem_iface = hmem_iface;
            fabric_state->world_size      = this->n_core;
            fabric_state->rank            = this->rank;

            init_fabric(fabric_state);
            if (!fabric_state->info)
                throw std::runtime_error("init_fabric failed: no suitable fabric found");
            if (hmem_iface != 0 && !is_hmem_capable(fabric_state))
                throw std::runtime_error(
                    "GPU source buffer requires DDSTORE_FABRIC=cxi "
                    "(current fabric does not support FI_HMEM)");

            /* Register the send buffer as an MR before writing the record --
             * same fi_mr_reg-vs-fi_mr_regattr branch as handshake(),
             * duplicated here the same way the host-only version already
             * is (method=2 registers inline instead of via handshake()). */
            bool send_is_hmem = hmem_iface != 0;
            int mr_rc;
            if (send_is_hmem)
            {
                struct iovec iov = {fabric_state->send_data, fabric_state->send_data_len};
                struct fi_mr_attr attr;
                memset(&attr, 0, sizeof(attr));
                attr.mr_iov    = &iov;
                attr.iov_count = 1;
                attr.access    = FI_WRITE | FI_REMOTE_READ;
                attr.iface     = (enum fi_hmem_iface)hmem_iface;
                attr.device.reserved = 0;
                mr_rc = fi_mr_regattr(fabric_state->domain, &attr, 0, &fabric_state->mr);
            }
            else
            {
                mr_rc = fi_mr_reg(
                    fabric_state->domain,
                    fabric_state->send_data,
                    fabric_state->send_data_len,
                    FI_WRITE | FI_REMOTE_READ,
                    0, 0, 0,
                    &fabric_state->mr,
                    NULL);
            }
            if (mr_rc != FI_SUCCESS)
                throw std::runtime_error(
                    std::string(send_is_hmem ? "fi_mr_regattr failed: " : "fi_mr_reg failed: ")
                    + fi_strerror(mr_rc));

            /* CXI (FI_MR_ENDPOINT): bind MR to endpoint and enable it before
             * use. The provider-assigned key is only valid after
             * fi_mr_enable() — same requirement as method=1's handshake().
             * No-op for hsn (is_mr_endpoint() is false).                     */
            if (is_mr_endpoint(fabric_state))
            {
                int rc = fi_mr_bind(fabric_state->mr, &fabric_state->signal->fid, 0);
                if (rc != FI_SUCCESS)
                    throw std::runtime_error(std::string("fi_mr_bind failed: ") + fi_strerror(rc));
                rc = fi_mr_enable(fabric_state->mr);
                if (rc != FI_SUCCESS)
                    throw std::runtime_error(std::string("fi_mr_enable failed: ") + fi_strerror(rc));
            }
            fabric_state->key = fi_mr_key(fabric_state->mr);

            /* Exchange records with all core ranks via MPI_Allgather, and
             * (rank 0 only) publish the combined record set for extra
             * members to join later. */
            std::vector<long> raw_lens(this->n_core);
            if (handshake_write(fabric_state, this->comm,
                                this->handshake_dir.c_str(), name.c_str(),
                                this->n_core, nrows, disp, (int)sizeof(T),
                                raw_lens.data()) != 0)
                throw std::runtime_error("handshake_write failed");

            /* Build prefix-sum lenlist from the raw per-rank row counts. */
            long sum = 0;
            std::vector<long> lenlist(this->n_core);
            for (int i = 0; i < this->n_core; i++)
            {
                sum += raw_lens[i];
                lenlist[i] = sum;
            }

            VarInfo_t var;
            var.name         = name;
            var.itemsize     = (int)sizeof(T);
            var.disp         = disp;
            var.win          = MPI_WIN_NULL;
            var.lenlist      = lenlist;
            var.active       = true;
            var.fence_active = false;
            var.base         = base;
            var.owns_base    = (hmem_iface == 0);
            var.fabric_state = fabric_state;
            this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
            return; /* lenlist already stored; skip the MPI_Allgather block below */
        }

        std::vector<long> lenlist(this->comm_size);
        MPI_Allgather(&nrows, 1, MPI_LONG, lenlist.data(), 1, MPI_LONG, this->comm);

        int max_disp = 0;
        // We assume disp is same for all
        MPI_Allreduce(&disp, &max_disp, 1, MPI_INT, MPI_MAX, this->comm);
        if (max_disp != disp)
            throw std::invalid_argument("Invalid disp");

        long sum = 0;
        for (long unsigned int i = 0; i < lenlist.size(); i++)
        {
            sum += lenlist[i];
            lenlist[i] = sum;
        }

        VarInfo_t var;
        var.name = name;
        var.itemsize = sizeof(T);
        var.disp = disp;
        var.win = win;
        var.lenlist = lenlist;
        var.active = true;
        var.fence_active = false;
        var.base = base;
        var.owns_base = (hmem_iface == 0);
        var.fabric_state = fabric_state;

        this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
    }

    void init(std::string name, long nrows, int disp, int itemsize)
    {
        void *base = NULL;
        // std::cout << "Init: " << name << ", nrows: " << nrows << ", disp: " << disp << ", itemsize: " << itemsize << std::endl;
        // std::cout << "Pre-allocating memory: " << (nrows * disp * itemsize)/1024/1024/1024 << " GB" << std::endl;
        int err = MPI_Alloc_mem((MPI_Aint)(nrows * disp * itemsize), MPI_INFO_NULL, &base);
        if (err)
        {
            exit(1);
        }
        memset(base, 0, nrows * disp * itemsize);

        MPI_Win win = MPI_WIN_NULL;
        struct fabric_state *fabric_state = NULL;

        if (this->method == 0)
        {
            MPI_Win_create(base,                               /* pre-allocated buffer */
                           (MPI_Aint)nrows * disp * itemsize, /* size in bytes */
                           disp * itemsize,                   /* displacement units */
                           MPI_INFO_NULL,                      /* info object */
                           this->comm,                         /* communicator */
                           &win /* window object */);
        }
        else if (this->method == 1)
        {
            fabric_state = (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
            pthread_mutex_init(&fabric_state->recv_lock, NULL);
            fabric_state->send_data = (char *)base;
            fabric_state->send_data_len = nrows * disp * itemsize;
            fabric_state->world_size = this->comm_size;
            fabric_state->rank = this->rank;

            init_fabric(fabric_state);
            if (!fabric_state->info)
                throw std::runtime_error("init_fabric failed: no suitable fabric found");
            if (handshake(fabric_state, this->comm) != 0)
                throw std::runtime_error("handshake failed (method=1)");
        }
        else if (this->method == 2)
        {
            fabric_state = (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
            pthread_mutex_init(&fabric_state->recv_lock, NULL);
            fabric_state->send_data     = (char *)base;
            fabric_state->send_data_len = nrows * disp * itemsize;
            fabric_state->world_size    = this->n_core;
            fabric_state->rank          = this->rank;

            init_fabric(fabric_state);
            if (!fabric_state->info)
                throw std::runtime_error("init_fabric failed: no suitable fabric found");

            int mr_rc = fi_mr_reg(
                fabric_state->domain,
                fabric_state->send_data,
                fabric_state->send_data_len,
                FI_WRITE | FI_REMOTE_READ,
                0, 0, 0,
                &fabric_state->mr,
                NULL);
            if (mr_rc != FI_SUCCESS)
                throw std::runtime_error(std::string("fi_mr_reg failed: ") + fi_strerror(mr_rc));

            /* CXI (FI_MR_ENDPOINT): bind MR to endpoint and enable it before
             * use. The provider-assigned key is only valid after
             * fi_mr_enable() — same requirement as method=1's handshake().
             * No-op for hsn (is_mr_endpoint() is false).                     */
            if (is_mr_endpoint(fabric_state))
            {
                int rc = fi_mr_bind(fabric_state->mr, &fabric_state->signal->fid, 0);
                if (rc != FI_SUCCESS)
                    throw std::runtime_error(std::string("fi_mr_bind failed: ") + fi_strerror(rc));
                rc = fi_mr_enable(fabric_state->mr);
                if (rc != FI_SUCCESS)
                    throw std::runtime_error(std::string("fi_mr_enable failed: ") + fi_strerror(rc));
            }
            fabric_state->key = fi_mr_key(fabric_state->mr);

            std::vector<long> raw_lens(this->n_core);
            if (handshake_write(fabric_state, this->comm,
                                this->handshake_dir.c_str(), name.c_str(),
                                this->n_core, nrows, disp, itemsize,
                                raw_lens.data()) != 0)
                throw std::runtime_error("handshake_write failed");

            long sum = 0;
            std::vector<long> lenlist(this->n_core);
            for (int i = 0; i < this->n_core; i++)
            {
                sum += raw_lens[i];
                lenlist[i] = sum;
            }

            VarInfo_t var;
            var.name         = name;
            var.itemsize     = itemsize;
            var.disp         = disp;
            var.win          = MPI_WIN_NULL;
            var.lenlist      = lenlist;
            var.active       = true;
            var.fence_active = false;
            var.base         = base;
            var.owns_base    = true;
            var.fabric_state = fabric_state;
            this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
            return;
        }

        std::vector<long> lenlist(this->comm_size);
        MPI_Allgather(&nrows, 1, MPI_LONG, lenlist.data(), 1, MPI_LONG, this->comm);

        int max_disp = 0;
        // We assume disp is same for all
        MPI_Allreduce(&disp, &max_disp, 1, MPI_INT, MPI_MAX, this->comm);
        if (max_disp != disp)
            throw std::invalid_argument("Invalid disp");

        long sum = 0;
        for (long unsigned int i = 0; i < lenlist.size(); i++)
        {
            sum += lenlist[i];
            lenlist[i] = sum;
        }

        VarInfo_t var;
        var.name = name;
        var.itemsize = itemsize;
        var.disp = disp;
        var.win = win;
        var.lenlist = lenlist;
        var.active = true;
        var.fence_active = false;
        var.base = base;
        var.owns_base = true;
        var.fabric_state = fabric_state;

        this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
    }

    template <typename T>
    void update(std::string name, T *buffer, long nrows, long offset = 0)
    {
        const VarInfo_t& varinfo = this->varlist.at(name);

        void *base = varinfo.base;
        int itemsize = varinfo.itemsize;
        int disp = varinfo.disp;
        if (itemsize != sizeof(T))
            throw std::invalid_argument("Invalid data type");

        memcpy((char*)base + offset * disp * itemsize, buffer, nrows * disp * itemsize);
    }

    /* hmem_iface: 0 (FI_HMEM_SYSTEM) for a host buffer, or an fi_hmem_iface
     * value (FI_HMEM_CUDA, FI_HMEM_ROCR, ...) identifying what kind of GPU
     * memory `buffer` is. Left as a plain int (not the enum) so the Cython
     * binding (pyddstore/_core.pyx) can pass it without cimporting the enum;
     * read_from_remote() in common.cxx casts it back before use.               */
    template <typename T>
    void get(std::string name, long start, long count, T *buffer, int hmem_iface = 0)
    {
        const VarInfo_t& varinfo = this->varlist.at(name);

        if (varinfo.itemsize != sizeof(T))
            throw std::invalid_argument("Invalid data type");

        int target = sortedsearch(varinfo.lenlist, start);
        long offset = target > 0 ? varinfo.lenlist[target - 1] : 0;
        // std::cout << "target,offset,start,count: " << target << "," << offset << "," << start << "," << count <<
        // std::endl;

        if (start < offset)
            throw std::invalid_argument("Invalid start on target");

        if ((start + count) > varinfo.lenlist[target])
            throw std::invalid_argument("Invalid count on target");

        // std::cout << "target,offset,start,count: " << target << "," << offset << "," << start << "," << count <<
        // std::endl;

        if (this->method == 0 && hmem_iface != 0)
        {
            throw std::runtime_error("GPU destination buffer is not supported with method=0 (MPI_Win)");
        }
        else if (this->method == 0)
        {
            MPI_Win win = varinfo.win;
            MPI_Win_lock(MPI_LOCK_SHARED, target, 0, win);
            /*
            int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype
                        origin_datatype, int target_rank, MPI_Aint target_disp,
                        int target_count, MPI_Datatype target_datatype, MPI_Win
                        win)
            */
            MPI_Get(buffer,                           /* pre-allocated buffer on RMA origin process */
                    varinfo.disp * varinfo.itemsize * count, /* count on RMA origin process */
                    MPI_BYTE,                         /* type on RMA origin process */
                    target,                           /* rank of RMA target process */
                    start - offset,                   /* displacement on RMA target process */
                    varinfo.disp * varinfo.itemsize * count, /* count on RMA target process */
                    MPI_BYTE,                         /* type on RMA target process */
                    win /* window object */);
            MPI_Win_unlock(target, win);
        }
        else if (this->method == 1 || this->method == 2)
        {
            /* Methods 1 and 2 both use libfabric fi_read — same path.
             * Locked for the whole branch: the recv_data/recv_data_len/
             * recv_hmem_iface writes below are themselves racy across
             * concurrent get() calls on this variable, not just the
             * read_from_remote() call that follows them -- see recv_lock's
             * comment in common.h. */
            const bool prof = ddstore_profile_enabled();
            uint64_t t_wait = prof ? ddstore_now_ns() : 0;
            fabric_state_lock_guard lock(varinfo.fabric_state);
            if (prof)
            {
                varinfo.fabric_state->prof_calls++;
                varinfo.fabric_state->prof_lock_wait_ns += ddstore_now_ns() - t_wait;
            }
            if (hmem_iface != 0 && !is_hmem_capable(varinfo.fabric_state))
                throw std::runtime_error(
                    "GPU destination buffer requires DDSTORE_FABRIC=cxi "
                    "(current fabric does not support FI_HMEM)");
            varinfo.fabric_state->recv_data = (char *)buffer;
            varinfo.fabric_state->recv_data_len = (size_t)varinfo.disp * varinfo.itemsize * count;
            varinfo.fabric_state->recv_hmem_iface = hmem_iface;
            int rc = read_from_remote(varinfo.fabric_state, target, (start - offset) * varinfo.disp * varinfo.itemsize);
            if (rc != 0)
                throw std::runtime_error(
                    "read_from_remote failed with code " + std::to_string(rc) +
                    " (target=" + std::to_string(target) + ")");
        }
    }

    /* Register `len` bytes at `buffer` once as a destination for get() /
     * get_batch() of `name` (hmem_iface as in get()), so reads into it or
     * any part of it skip memory registration. Several buffers can be
     * registered per variable (e.g. one per loader thread); none is ever
     * evicted. The caller keeps the buffer alive until unregister_recv() or
     * free(). No-op for method 0.                                          */
    void register_recv(std::string name, void *buffer, size_t len, int hmem_iface = 0)
    {
        const VarInfo_t &varinfo = this->varlist.at(name);
        if (this->method == 0)
            return;
        if (hmem_iface != 0 && !is_hmem_capable(varinfo.fabric_state))
            throw std::runtime_error(
                "GPU destination buffer requires DDSTORE_FABRIC=cxi "
                "(current fabric does not support FI_HMEM)");
        fabric_state_lock_guard lock(varinfo.fabric_state);
        if (register_recv_region(varinfo.fabric_state, (char *)buffer, len, hmem_iface) != 0)
            throw std::runtime_error("register_recv failed for " + name);
    }

    /* Undo register_recv() for the buffer starting at `buffer`.            */
    void unregister_recv(std::string name, void *buffer)
    {
        const VarInfo_t &varinfo = this->varlist.at(name);
        if (this->method == 0)
            return;
        fabric_state_lock_guard lock(varinfo.fabric_state);
        if (unregister_recv_region(varinfo.fabric_state, (char *)buffer) != 0)
            throw std::invalid_argument("buffer is not registered for " + name);
    }

    /* Batched get: row i of `buffer` (n contiguous rows) receives global
     * row idx[i]; rows may come from any ranks, in any order, with repeats.
     * Every index is validated before anything is read. hmem_iface as in get().
     *
     * Methods 1/2 (one-sided): take the variable's lock once and post all n
     * fi_read()s before waiting for any (read_batch_from_remote()).
     *
     * Method 0 (collective, MDLoader-style): COLLECTIVE over this store's
     * communicator — every rank must call get_batch() for the same variable
     * the same number of times, in the same order (n may differ per rank,
     * including 0). See get_batch_alltoall().                              */
    template <typename T>
    void get_batch(std::string name, const long *idx, long n, T *buffer, int hmem_iface = 0)
    {
        const VarInfo_t& varinfo = this->varlist.at(name);

        if (varinfo.itemsize != sizeof(T))
            throw std::invalid_argument("Invalid data type");

        if (this->method == 0)
        {
            if (hmem_iface != 0)
                throw std::runtime_error("GPU destination buffer is not supported with method=0 (MPI_Win)");
            this->get_batch_alltoall(varinfo, idx, n, (char *)buffer);
            return;
        }

        if (n <= 0)
            return;

        size_t row_bytes = (size_t)varinfo.disp * varinfo.itemsize;
        std::vector<int> target(n);
        std::vector<uint64_t> offset(n);
        for (long i = 0; i < n; i++)
        {
            int t = sortedsearch(varinfo.lenlist, idx[i]); /* throws if out of range */
            long first = t > 0 ? varinfo.lenlist[t - 1] : 0;
            target[i] = t;
            offset[i] = (uint64_t)(idx[i] - first) * row_bytes;
        }

        /* Methods 1 and 2: one lock acquisition for the whole batch. */
        const bool prof = ddstore_profile_enabled();
        uint64_t t_wait = prof ? ddstore_now_ns() : 0;
        fabric_state_lock_guard lock(varinfo.fabric_state);
        if (prof)
        {
            varinfo.fabric_state->prof_calls++;
            varinfo.fabric_state->prof_lock_wait_ns += ddstore_now_ns() - t_wait;
        }
        if (hmem_iface != 0 && !is_hmem_capable(varinfo.fabric_state))
            throw std::runtime_error(
                "GPU destination buffer requires DDSTORE_FABRIC=cxi "
                "(current fabric does not support FI_HMEM)");
        varinfo.fabric_state->recv_data = (char *)buffer;
        varinfo.fabric_state->recv_data_len = n * row_bytes;
        varinfo.fabric_state->recv_hmem_iface = hmem_iface;
        int rc = read_batch_from_remote(varinfo.fabric_state, n, target.data(),
                                        offset.data(), row_bytes);
        if (rc != 0)
            throw std::runtime_error(
                "read_batch_from_remote failed with code " + std::to_string(rc) +
                " (" + std::to_string(n) + " rows)");
    }

private:
    /* Method 0 batched get, after MDLoader's collective module (Bae et al.,
     * IPDPSW 2024): every rank all-gathers the batch indices of all ranks,
     * packs the rows it owns for each requester, and one MPI_Alltoallv
     * delivers them; each rank then puts its rows in request order. Uses a
     * private duplicate of the store's communicator (coll_comm), so it never
     * matches the caller's own collectives or the windows' fences. The
     * indices are validated after the gather, on the global list, so every
     * rank throws the same error together instead of one rank leaving the
     * others blocked in the exchange.                                       */
    void get_batch_alltoall(const VarInfo_t &varinfo, const long *idx, long n, char *out)
    {
        std::lock_guard<std::mutex> guard(this->coll_mutex);
        if (this->coll_comm == MPI_COMM_NULL)
            MPI_Comm_dup(this->comm, &this->coll_comm);

        const int P = this->comm_size;
        const int me = this->rank;
        const size_t row = (size_t)varinfo.disp * varinfo.itemsize;

        /* 1. Every rank learns every rank's requests. */
        int nloc = (int)n;
        std::vector<int> nreq(P), rbase(P + 1, 0);
        MPI_Allgather(&nloc, 1, MPI_INT, nreq.data(), 1, MPI_INT, this->coll_comm);
        for (int p = 0; p < P; p++)
            rbase[p + 1] = rbase[p] + nreq[p];
        std::vector<long> all(rbase[P] > 0 ? rbase[P] : 1);
        MPI_Allgatherv(idx, nloc, MPI_LONG, all.data(), nreq.data(), rbase.data(),
                       MPI_LONG, this->coll_comm);

        const long total_rows = varinfo.lenlist.empty() ? 0 : varinfo.lenlist.back();
        for (int j = 0; j < rbase[P]; j++)
            if (all[j] < 0 || all[j] >= total_rows)
                throw std::out_of_range(
                    "Global index " + std::to_string(all[j]) +
                    " is out of range [0, " + std::to_string(total_rows) + ")");

        /* 2. Split the exchange into rounds of at most `cap` bytes received
         * per rank (DDSTORE_ALLTOALL_MAX_BYTES, default 2 MiB): one huge
         * Alltoallv of large rows is slower than per-row reads. Every rank
         * derives the same round count from the gathered request counts;
         * round k moves each rank's k-th slice of `per` requests.          */
        const long cap = alltoall_max_bytes();
        const long per = row >= (size_t)cap ? 1 : (long)(cap / (long)row);
        long max_req = 0;
        for (int p = 0; p < P; p++)
            max_req = nreq[p] > max_req ? nreq[p] : max_req;
        const long rounds = (max_req + per - 1) / per;

        const long my_first = me > 0 ? varinfo.lenlist[me - 1] : 0;
        const long my_end = varinfo.lenlist[me];
        std::vector<int> owner(n > 0 ? n : 1);
        for (long i = 0; i < n; i++)
            owner[i] = sortedsearch(varinfo.lenlist, idx[i]);

        MPI_Datatype rowtype;
        MPI_Type_contiguous((int)row, MPI_BYTE, &rowtype);
        MPI_Type_commit(&rowtype);
        std::vector<int> scount(P), sdispl(P), rcount(P), rdispl(P), next(P);
        std::vector<char> sendbuf, recvbuf;
        for (long k = 0; k < rounds; k++)
        {
            /* Rows this rank owns from every requester's slice, packed per
             * requester in request order. */
            for (int p = 0; p < P; p++)
            {
                scount[p] = 0;
                long lo = rbase[p] + k * per, hi = rbase[p] + std::min((long)nreq[p], (k + 1) * per);
                for (long j = lo; j < hi; j++)
                    if (all[j] >= my_first && all[j] < my_end)
                        scount[p]++;
            }
            sdispl[0] = 0;
            for (int p = 1; p < P; p++)
                sdispl[p] = sdispl[p - 1] + scount[p - 1];
            const long nsend = sdispl[P - 1] + scount[P - 1];
            sendbuf.resize((size_t)(nsend > 0 ? nsend : 1) * row);
            size_t ks = 0;
            for (int p = 0; p < P; p++)
            {
                long lo = rbase[p] + k * per, hi = rbase[p] + std::min((long)nreq[p], (k + 1) * per);
                for (long j = lo; j < hi; j++)
                    if (all[j] >= my_first && all[j] < my_end)
                        memcpy(sendbuf.data() + (ks++) * row,
                               (char *)varinfo.base + (size_t)(all[j] - my_first) * row, row);
            }

            /* Where this rank's own slice comes from. */
            const long ilo = std::min(n, k * per), ihi = std::min(n, (k + 1) * per);
            std::fill(rcount.begin(), rcount.end(), 0);
            for (long i = ilo; i < ihi; i++)
                rcount[owner[i]]++;
            rdispl[0] = 0;
            for (int p = 1; p < P; p++)
                rdispl[p] = rdispl[p - 1] + rcount[p - 1];
            recvbuf.resize((size_t)(ihi > ilo ? ihi - ilo : 1) * row);

            MPI_Alltoallv(sendbuf.data(), scount.data(), sdispl.data(), rowtype,
                          recvbuf.data(), rcount.data(), rdispl.data(), rowtype, this->coll_comm);

            /* Received rows are grouped by owner, each group in request
             * order: put them back in this rank's request order. */
            next = rdispl;
            for (long i = ilo; i < ihi; i++)
                memcpy(out + (size_t)i * row, recvbuf.data() + (size_t)(next[owner[i]]++) * row, row);
        }
        MPI_Type_free(&rowtype);
    }

    /* DDSTORE_ALLTOALL_MAX_BYTES: method 0 get_batch() bytes received per
     * rank per exchange round (default 2 MiB; read once). Must be the same
     * on every rank: the round count is derived from it.                   */
    static long alltoall_max_bytes()
    {
        static long cap = -1;
        if (cap < 0)
        {
            const char *e = getenv("DDSTORE_ALLTOALL_MAX_BYTES");
            long v = e ? atol(e) : 0;
            cap = v > 0 ? v : 2L * 1024 * 1024;
        }
        return cap;
    }

    int method; // 0: MPI, 1: libfabric, 2: file-based handshake (libfabric transport)
    MPI_Comm   coll_comm = MPI_COMM_NULL; /* method 0 get_batch; see above  */
    std::mutex coll_mutex;                /* one get_batch_alltoall at a time */

    MPI_Comm    comm;
    int         comm_size;
    int         rank;

    /* Method 2 fields */
    std::string handshake_dir;  /* shared directory for CoreRecord files     */
    int         n_core;         /* number of core ranks                      */
    bool        is_extra;       /* true if this is an extra (read-only) node */

    std::unordered_map<std::string, VarInfo_t> varlist;
};
