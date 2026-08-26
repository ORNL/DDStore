#include <iostream>
#include <cstring>
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

    template <typename T>
    void add(std::string name, T *buffer, long nrows, int disp)
    {
        void *base = NULL;
        // (2025/03) jyc: necessary to avoid memory error
        int err = MPI_Alloc_mem((MPI_Aint)(nrows * disp * sizeof(T)), MPI_INFO_NULL, &base);
        if (err)
        {
            exit(1);
        }
        memcpy(base, buffer, nrows * disp * sizeof(T));

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
            fabric_state->send_data = (char *)base;
            fabric_state->send_data_len = nrows * disp * sizeof(T);
            fabric_state->world_size = this->comm_size;
            fabric_state->rank = this->rank;

            init_fabric(fabric_state);
            if (!fabric_state->info)
                throw std::runtime_error("init_fabric failed: no suitable fabric found");
            handshake(fabric_state, this->comm);
        }
        else if (this->method == 2)
        {
            fabric_state = (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
            fabric_state->send_data     = (char *)base;
            fabric_state->send_data_len = nrows * disp * sizeof(T);
            fabric_state->world_size    = this->n_core;
            fabric_state->rank          = this->rank;

            init_fabric(fabric_state);
            if (!fabric_state->info)
                throw std::runtime_error("init_fabric failed: no suitable fabric found");

            /* Register the send buffer as an MR before writing the record. */
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
            fabric_state->send_data = (char *)base;
            fabric_state->send_data_len = nrows * disp * itemsize;
            fabric_state->world_size = this->comm_size;
            fabric_state->rank = this->rank;

            init_fabric(fabric_state);
            if (!fabric_state->info)
                throw std::runtime_error("init_fabric failed: no suitable fabric found");
            handshake(fabric_state, this->comm);
        }
        else if (this->method == 2)
        {
            fabric_state = (struct fabric_state *)calloc(1, sizeof(struct fabric_state));
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

    template <typename T>
    void get(std::string name, long start, long count, T *buffer)
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

        if (this->method == 0)
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
            /* Methods 1 and 2 both use libfabric fi_read — same path. */
            varinfo.fabric_state->recv_data = (char *)buffer;
            varinfo.fabric_state->recv_data_len = varinfo.disp * varinfo.itemsize * count;
            read_from_remote(varinfo.fabric_state, target, (start - offset) * varinfo.disp * varinfo.itemsize);
        }
    }

private:
    int method; // 0: MPI, 1: libfabric, 2: file-based handshake (libfabric transport)

    MPI_Comm    comm;
    int         comm_size;
    int         rank;

    /* Method 2 fields */
    std::string handshake_dir;  /* shared directory for CoreRecord files     */
    int         n_core;         /* number of core ranks                      */
    bool        is_extra;       /* true if this is an extra (read-only) node */

    std::unordered_map<std::string, VarInfo_t> varlist;
};
