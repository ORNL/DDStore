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
 * Method 2: file-based handshake record, one per core rank.
 * All n_core records for a variable are gathered in memory (via MPI among
 * core ranks) and published as a single combined file, written once by
 * core rank 0:
 *   {handshake_dir}/{varname}.bin
 * ----------------------------------------------------------------------- */
struct CoreRecord
{
    char     fabric_address[DP_AV_DEF_SIZE]; /* raw fi_getname output          */
    size_t   fabric_address_len;             /* actual bytes used               */
    uint64_t key;                            /* MR key from fi_mr_key()         */
    uint64_t base_address;                   /* virtual address of send_data    */
    long     nrows;                          /* rows owned by this core rank    */
    int      disp;                           /* elements per row                */
    int      itemsize;                       /* bytes per element               */
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
     * CXI, so we detect by provider name instead of mr_mode flags. False
     * (no-op) for every provider dev-file2 already supports (hsn/verbs/
     * gni/psm2), since none of those set mr_mode & FI_MR_ENDPOINT and none
     * are named "cxi".                                                       */
    static bool is_mr_endpoint(struct fabric_state *f)
    {
        return (f->info->domain_attr->mr_mode & FI_MR_ENDPOINT) != 0 ||
               (f->info->fabric_attr->prov_name &&
                strcmp(f->info->fabric_attr->prov_name, "cxi") == 0);
    }

    /* With FI_MR_VIRT_ADDR the fi_read remote addr is the virtual address.
     * CXI does NOT use virtual addresses — offset is 0-based from MR base.
     *
     * NOTE: this is deliberately NOT a mr_mode bit check. dev-file2's
     * init_fabric_hsn() sets mr_mode to the legacy FI_MR_BASIC sentinel,
     * which on this system's libfabric (2.3.1) is bit 0 (value 1) — a
     * completely different bit than FI_MR_VIRT_ADDR (bit 4). A `mr_mode &
     * FI_MR_VIRT_ADDR` check would therefore silently resolve to false for
     * hsn, breaking address exchange for the already-proven path. Before
     * this helper existed, dev-file2 unconditionally used the real pointer
     * for every provider it supported (hsn/verbs/gni/psm2) — no virt-addr/
     * prov-key distinction existed at all — so preserve that unconditional
     * behavior for anything that isn't cxi.                                  */
    static bool is_virt_addr(struct fabric_state *f)
    {
        return !(f->info->fabric_attr->prov_name &&
                 strcmp(f->info->fabric_attr->prov_name, "cxi") == 0);
    }

    void init_fabric(struct fabric_state *fabric);
    int handshake(struct fabric_state *fabric_state, MPI_Comm comm);
    int read_from_remote(struct fabric_state *fabric_state, int src, uint64_t offset);

    /* --- Method 2: file-based handshake ---------------------------------- */

    /* Resolve the handshake directory (priority: user_dir > env var > cwd).
     * Creates the directory if it does not exist.
     * Returns pointer to a static buffer — copy if needed across calls.     */
    const char *resolve_handshake_dir(const char *user_dir);

    /* Core rank: exchange CoreRecords with all other core ranks via
     * MPI_Allgather over `comm` (no filesystem round-trip needed for
     * core-to-core discovery), populate this rank's fs->comm_partner[],
     * remote_key[], remote_address[], and fill lenlist[0..n_core-1] (raw
     * row counts, NOT yet prefix-summed).  Rank 0 additionally publishes
     * the combined record set to {dir}/{varname}.bin (tmp + fsync + rename)
     * so extra members can join later.                                       */
    int handshake_write(struct fabric_state *fs, MPI_Comm comm,
                        const char *dir, const char *varname,
                        int n_core, long nrows, int disp, int itemsize,
                        long *lenlist);

    /* Extra member: poll for {dir}/{varname}.bin (single file holding all
     * n_core CoreRecords), read it, and populate this process's
     * fs->comm_partner[], remote_key[], remote_address[], and
     * lenlist[0..n_core-1] (raw row counts, NOT yet prefix-summed).  Does
     * NOT write anything.  Blocks until the file appears (with timeout).     */
    int handshake_join(struct fabric_state *fs,
                       const char *dir, const char *varname,
                       int n_core,
                       long *lenlist, int *out_disp, int *out_itemsize);

#ifdef __cplusplus
}
#endif
