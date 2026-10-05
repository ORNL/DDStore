#pragma once

#include "rdma/fabric.h"
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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

    /* One caller-registered recv buffer (see fabric_state::pinned). */
    struct recv_region
    {
        struct fid_mr *mr;
        char *base;
        size_t len;
        int hmem_iface;
    };

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
        /* FI_HMEM_SYSTEM (0) if send_data is host memory; otherwise the
         * fi_hmem_iface value identifying what kind of GPU memory it is.
         * Set by the caller (see ddstore.hpp's add()) and forwarded into
         * fi_mr_regattr()'s attr.iface in handshake() (method=1) / add()'s
         * inline registration (method=2). A separate field from
         * recv_hmem_iface: one fabric_state can be simultaneously the send
         * side (registered once at add() time) and the recv side
         * (re-registered per get() call, including self-reads) -- these
         * are independent MRs with independent lifetimes.                    */
        int send_hmem_iface;
        char *recv_data;
        size_t recv_data_len;
        /* FI_HMEM_SYSTEM (0) if recv_data is host memory; otherwise the
         * fi_hmem_iface value (FI_HMEM_CUDA, FI_HMEM_ROCR, ...) identifying
         * what kind of GPU memory it is. Set by the caller (see
         * ddstore.hpp's get()); the value is opaque here, just forwarded
         * into fi_mr_regattr()'s attr.iface in read_from_remote().           */
        int recv_hmem_iface;
        struct fid_mr *mr;
        struct fid_mr *recv_mr;
        /* Cached recv-side MR region: the registered range is
         * [recv_mr_base, recv_mr_base + recv_mr_reg_len).  Any recv_data
         * pointer that falls within this range with recv_data_len bytes
         * fitting inside it can reuse recv_mr without re-registration.
         *
         * Hits when the same buffer (or a sub-range of it) is passed again --
         * e.g. PyTorch's caching allocator returning the same block for a
         * same-shape torch.empty() -- so get() doesn't re-register every call.
         *
         * Initialised to NULL/0 so the first call always registers.          */
        char  *recv_mr_base;
        size_t recv_mr_reg_len;
        /* Caller-registered recv regions (register_recv_region()), checked
         * before the one-slot cache above and never evicted: the caller owns
         * these buffers and keeps them alive until unregister / free().     */
        struct recv_region *pinned;
        int n_pinned;
        /* Largest single fi_read() the endpoint accepts (FI_OPT_MAX_MSG_SIZE
         * or ep_attr->max_msg_size); 0 if unknown. Longer rows are split.   */
        size_t max_msg_size;
        uint64_t key;
        uint64_t *remote_key;
        uint64_t *remote_address;

        int world_size;
        int rank;

        /* Serializes concurrent get() calls on THIS variable's
         * fabric_state -- see fabric_state_lock_guard below. libfabric
         * itself doesn't guarantee thread safety unless the domain is
         * opened with FI_THREAD_SAFE (it isn't here; see
         * init_fabric_hsn()'s FI_THREAD_DOMAIN hint and init_fabric_cxi()'s
         * unconstrained NULL-hints query), and even then that would only
         * cover libfabric's own objects, not the plain fields above
         * (recv_data/recv_mr/recv_mr_base/recv_mr_reg_len) that read_from_
         * remote() reads and writes as a cache.
         * Confirmed necessary by direct experiment: concurrent get() calls
         * without this crashed with "double free or corruption".
         * Zero-initialized by calloc() at every allocation site below, but
         * explicitly pthread_mutex_init()'d right after each one anyway --
         * relying on zero-initialized pthread_mutex_t being equivalent to
         * PTHREAD_MUTEX_INITIALIZER is a common but implementation-defined
         * assumption; init explicitly instead. */
        pthread_mutex_t recv_lock;

        /* DDSTORE_PROFILE=1: cumulative get() timing for this variable, all
         * updated while recv_lock is held (see ddstore_profile_enabled()).
         * prof_lock_wait_ns is the time spent waiting to acquire recv_lock;
         * mr = recv-MR cache check / (re)registration; read = posting
         * fi_read(); cq = polling the CQ until the read completes.          */
        uint64_t prof_calls;  /* get() / get_batch() calls              */
        uint64_t prof_rows;   /* rows read by those calls               */
        uint64_t prof_lock_wait_ns;
        uint64_t prof_mr_ns;
        uint64_t prof_mr_miss;
        uint64_t prof_read_ns;
        uint64_t prof_cq_ns;
    };

    static inline uint64_t ddstore_now_ns(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    }

    /* True if DDSTORE_PROFILE is set to a non-"0" value (read once).        */
    static inline bool ddstore_profile_enabled(void)
    {
        static int enabled = -1;
        if (enabled < 0)
        {
            const char *e = getenv("DDSTORE_PROFILE");
            enabled = (e && e[0] && strcmp(e, "0") != 0) ? 1 : 0;
        }
        return enabled == 1;
    }

    static bool is_local_mr_req(struct fabric_state *f)
    {
        return (f->info->mode & FI_LOCAL_MR) != 0;
    }

    /* CXI (and some other providers) use FI_MR_ENDPOINT: after fi_mr_reg the
     * MR must be bound to the endpoint and enabled before it can be used, and
     * the key is only valid after fi_mr_enable().
     * With NULL hints fi_getinfo returns mr_mode=0 even for CXI (seen on
     * Perlmutter), so we detect by provider name instead of mr_mode flags.
     * False (no-op) for hsn/verbs/gni/psm2, since none of those set
     * mr_mode & FI_MR_ENDPOINT and none are named "cxi".                  */
    static bool is_mr_endpoint(struct fabric_state *f)
    {
        return (f->info->domain_attr->mr_mode & FI_MR_ENDPOINT) != 0 ||
               (f->info->fabric_attr->prov_name &&
                strcmp(f->info->fabric_attr->prov_name, "cxi") == 0);
    }

    /* With FI_MR_VIRT_ADDR the fi_read remote addr is the virtual address.
     * CXI does NOT use virtual addresses — offset is 0-based from MR base.
     *
     * NOTE: this is deliberately NOT a mr_mode bit check. init_fabric_hsn()
     * sets mr_mode to the legacy FI_MR_BASIC sentinel,
     * which on this system's libfabric (2.3.1) is bit 0 (value 1) — a
     * completely different bit than FI_MR_VIRT_ADDR (bit 4). A `mr_mode &
     * FI_MR_VIRT_ADDR` check would therefore silently resolve to false for
     * hsn, breaking address exchange for the already-proven path. Before
     * cxi support, the real pointer was used unconditionally for every
     * provider (hsn/verbs/gni/psm2), so preserve that for anything that
     * isn't cxi.                                                         */
    static bool is_virt_addr(struct fabric_state *f)
    {
        return !(f->info->fabric_attr->prov_name &&
                 strcmp(f->info->fabric_attr->prov_name, "cxi") == 0);
    }

    /* True only for the cxi provider (the real Slingshot/HW path). GPU
     * (ROCr HMEM) buffer registration is only attempted when this is true —
     * verified empirically that cxi's NULL-hints fi_getinfo() already
     * reports FI_HMEM in caps by default on Frontier; hsn (tcp;ofi_rxm)
     * has no such support and would otherwise fail with a confusing
     * low-level libfabric error instead of a clear one.                      */
    static bool is_hmem_capable(struct fabric_state *f)
    {
        return f->info && f->info->fabric_attr->prov_name &&
               strcmp(f->info->fabric_attr->prov_name, "cxi") == 0;
    }

    void init_fabric(struct fabric_state *fabric);
    int handshake(struct fabric_state *fabric_state, MPI_Comm comm);
    int read_from_remote(struct fabric_state *fabric_state, int src, uint64_t offset);
    /* n rows of row_len bytes into recv_data (recv_data_len == n * row_len);
     * row i from rank src[i] at byte offset offset[i]. All reads are posted
     * before any is waited for. 0 on success. See common.cxx.             */
    int read_batch_from_remote(struct fabric_state *fabric_state, long n,
                               const int *src, const uint64_t *offset, size_t row_len);
    /* Register [base, base + len) once as a recv buffer (hmem_iface as for
     * recv_hmem_iface); reads into it then skip registration. Registering a
     * region already registered is a no-op. 0 on success. Caller holds
     * recv_lock and keeps the buffer alive until unregister / free.        */
    int register_recv_region(struct fabric_state *fs, char *base, size_t len, int hmem_iface);
    /* Undo register_recv_region() for the region starting at base. 0 on
     * success, 1 if no such region. Caller holds recv_lock.               */
    int unregister_recv_region(struct fabric_state *fs, char *base);
    /* Close every registered recv region (free()).                         */
    void close_recv_regions(struct fabric_state *fs);

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

/* RAII guard for struct fabric_state::recv_lock -- locks on construction,
 * unlocks on destruction (including when leaving via an exception), so
 * every exit path of the critical section it wraps is covered without
 * having to hand-place lock/unlock calls on each one. See the comment on
 * recv_lock above for what this protects and why. */
struct fabric_state_lock_guard
{
    struct fabric_state *fs;
    explicit fabric_state_lock_guard(struct fabric_state *fs) : fs(fs)
    {
        pthread_mutex_lock(&fs->recv_lock);
    }
    ~fabric_state_lock_guard()
    {
        pthread_mutex_unlock(&fs->recv_lock);
    }
    fabric_state_lock_guard(const fabric_state_lock_guard &) = delete;
    fabric_state_lock_guard &operator=(const fabric_state_lock_guard &) = delete;
};
#endif
