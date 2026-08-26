#pragma once

#include "rdma/fabric.h"
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <mpi.h>

#define DP_AV_DEF_SIZE 512
#define COMM_FILE_WRITER_TO_READER "./writer_address.bin"

/* -----------------------------------------------------------------------
 * Method 2: file-based handshake record written by each core rank.
 * One shared file per variable, with each core rank owning a fixed-size
 * slot (rank * sizeof(struct CoreRecord)) inside it:
 *   {handshake_dir}/{varname}.bin
 *
 * `ready` is the last field and is committed with its own pwrite()+fsync()
 * after the rest of the slot is written and fsync()'d (see handshake_write()
 * in common.cxx) -- since many independent processes write into disjoint
 * byte ranges of the same file, there is no per-rank file to rename into
 * place atomically, so readers instead poll each slot's `ready` scalar to
 * tell a fully-written record apart from one still being written (or a
 * sparse hole that was never written at all).
 * ----------------------------------------------------------------------- */
#define CORE_RECORD_READY_MAGIC 0xD57A2E01u

struct CoreRecord
{
    char     fabric_address[DP_AV_DEF_SIZE]; /* raw fi_getname output          */
    size_t   fabric_address_len;             /* actual bytes used               */
    uint64_t key;                            /* MR key from fi_mr_key()         */
    uint64_t base_address;                   /* virtual address of send_data    */
    long     nrows;                          /* rows owned by this core rank    */
    int      disp;                           /* elements per row                */
    int      itemsize;                       /* bytes per element               */
    uint32_t ready;                          /* CORE_RECORD_READY_MAGIC when the
                                               * rest of this slot is fully
                                               * written; must stay last         */
};

#ifdef __cplusplus
extern "C"
{
#endif

    struct fabric_state
    {
        struct fi_context *ctx;
        struct fi_info *info;
        struct fid_fabric *fabric;
        struct fid_domain *domain;
        struct fid_ep *signal;
        struct fid_cq *cq_signal;
        struct fid_av *av;

        fi_addr_t *comm_partner;
        char *send_data;
        size_t send_data_len;
        char *recv_data;
        size_t recv_data_len;
        struct fid_mr *mr;
        struct fid_mr *recv_mr;
        uint64_t key;
        uint64_t *remote_key;
        uint64_t *remote_address;

        int world_size;
        int rank;
    };

    static bool is_local_mr_req(struct fabric_state *f)
    {
        return (f->info->mode & FI_LOCAL_MR) != 0;
    }

    /* CXI (and some other providers) use FI_MR_ENDPOINT: after fi_mr_reg the
     * MR must be bound to the endpoint and enabled before it can be used, and
     * the key is only valid after fi_mr_enable().
     * On Perlmutter, fi_getinfo with NULL hints returns mr_mode=0 even for
     * CXI, so we detect by provider name instead of mr_mode flags.          */
    static bool is_mr_endpoint(struct fabric_state *f)
    {
        return (f->info->domain_attr->mr_mode & FI_MR_ENDPOINT) != 0 ||
               (f->info->fabric_attr->prov_name &&
                strcmp(f->info->fabric_attr->prov_name, "cxi") == 0);
    }

    /* With FI_MR_PROV_KEY the provider assigns the key.
     * CXI always uses provider-assigned keys.                                */
    static bool is_prov_key(struct fabric_state *f)
    {
        return (f->info->domain_attr->mr_mode & FI_MR_PROV_KEY) != 0 ||
               (f->info->fabric_attr->prov_name &&
                strcmp(f->info->fabric_attr->prov_name, "cxi") == 0);
    }

    /* With FI_MR_VIRT_ADDR the fi_read remote addr is the virtual address.
     * CXI does NOT use virtual addresses — offset is 0-based from MR base.  */
    static bool is_virt_addr(struct fabric_state *f)
    {
        if (f->info->fabric_attr->prov_name &&
            strcmp(f->info->fabric_attr->prov_name, "cxi") == 0)
            return false;
        return (f->info->domain_attr->mr_mode & FI_MR_VIRT_ADDR) != 0;
    }

    void init_fabric(struct fabric_state *fabric);
    int handshake(struct fabric_state *fabric_state, MPI_Comm comm);
    int read_from_remote(struct fabric_state *fabric_state, int src, uint64_t offset);

    /* --- Method 2: file-based handshake ---------------------------------- */

    /* Resolve the handshake directory (priority: user_dir > env var > cwd).
     * Creates the directory if it does not exist.
     * Returns pointer to a static buffer — copy if needed across calls.     */
    const char *resolve_handshake_dir(const char *user_dir);

    /* Core rank: write own CoreRecord into this rank's slot of the shared
     * {dir}/{varname}.bin file, committing `ready` last (see CoreRecord).    */
    int handshake_write(struct fabric_state *fs,
                        const char *dir, const char *varname, int rank,
                        long nrows, int disp, int itemsize);

    /* Core rank: poll the shared file until all n_core slots are ready, then
     * read them all and populate fs->comm_partner[], remote_key[],
     * remote_address[], and fill lenlist[0..n_core-1] (raw row counts,
     * NOT yet prefix-summed).  Also sets *out_disp and *out_itemsize.        */
    int handshake_read(struct fabric_state *fs,
                       const char *dir, const char *varname,
                       int n_core, int my_rank,
                       long *lenlist, int *out_disp, int *out_itemsize);

    /* Extra member: same as handshake_read but does NOT write anything.
     * Blocks until all n_core slots are ready (with timeout).                */
    int handshake_join(struct fabric_state *fs,
                       const char *dir, const char *varname,
                       int n_core,
                       long *lenlist, int *out_disp, int *out_itemsize);

#ifdef __cplusplus
}
#endif
