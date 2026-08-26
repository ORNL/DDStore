#pragma once

#include "rdma/fabric.h"
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
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
