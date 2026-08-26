#include "common.h"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

void init_fabric(struct fabric_state *fabric)
{
    struct fi_info *hints, *info, *originfo, *useinfo;
    struct fi_av_attr av_attr = {FI_AV_UNSPEC};
    struct fi_cq_attr cq_attr = {0};
    char *ifname;
    int result;

    ifname = getenv("FABRIC_IFACE");
    fabric->info = NULL;

    int version = fi_version();

    /* IMPORTANT: fi_getinfo() must be called with a literal NULL hints
     * pointer.  Passing ANY hints struct (even one with only ep_attr->type
     * set) causes fi_getinfo to return NULL inside this process (torch +
     * NCCL + mpi4py all loaded) on Perlmutter compute nodes, even though a
     * plain MPI-only C program does not exhibit this.  With a literal NULL
     * hints pointer, fi_getinfo returns fully-populated fi_info entries
     * (mr_mode, max_msg_size, tx_attr, rx_attr all correctly set) — verified
     * via a standalone test binary run under the same srun job.  Do NOT
     * "improve" this by adding hints back without re-verifying against a
     * real running job (not just fi_info on the login node).                */
    fi_getinfo(version, NULL, NULL, 0, NULL, &info);
    if (!info)
    {
        fprintf(stderr, "no fabrics detected.\n");
        return;
    }

    originfo = info;
    useinfo = NULL;
    while (info)
    {
        char *prov_name = info->fabric_attr->prov_name;
        char *domain_name = info->domain_attr->name;

        /* If FABRIC_IFACE is set, match by domain name against cxi or tcp;ofi_rxm. */
        if (ifname && domain_name && (strcmp(ifname, domain_name) == 0) &&
            (strcmp(prov_name, "cxi") == 0 || strcmp(prov_name, "tcp;ofi_rxm") == 0))
        {
            fprintf(stderr, "using interface set by FABRIC_IFACE: %s (%s).\n",
                    domain_name, prov_name);
            useinfo = info;
            break;
        }
        if ((strcmp(prov_name, "cxi") == 0 ||
             (strcmp(prov_name, "verbs") == 0 && info->src_addr) ||
             strcmp(prov_name, "gni") == 0 ||
             strcmp(prov_name, "psm2") == 0) &&
            (!useinfo || (ifname && domain_name &&
                          strcmp(useinfo->domain_attr->name, ifname) != 0)))
        {
            useinfo = info;
        }
        else
        {
            // fprintf(
            //     stderr,
            //     "ignoring fabric %s because it's not of a supported type. It "
            //     "may work to force this fabric to be used by setting "
            //     "FABRIC_IFACE to %s, but it may not be stable or performant.\n",
            //     prov_name,
            //     domain_name);
        }
        info = info->next;
    }

    info = useinfo;

    if (!info)
    {
        fprintf(
            stderr,
            "none of the usable system fabrics are supported high speed "
            "interfaces (verbs, gni, psm2.) To use a compatible fabric that is "
            "being ignored (probably sockets), set the environment variable "
            "FABRIC_IFACE to the interface name. Check the output of fi_info "
            "to troubleshoot this message.\n");
        fabric->info = NULL;
        return;
    }

    fabric->info = fi_dupinfo(info);
    fi_freeinfo(originfo);
    originfo = NULL;

open_fabric:
    info = fabric->info;

    if (info->mode & FI_CONTEXT2)
    {
        fabric->ctx = (fi_context*) calloc(2, sizeof(*fabric->ctx));
    }
    else if (info->mode & FI_CONTEXT)
    {
        fabric->ctx = (fi_context*) calloc(1, sizeof(*fabric->ctx));
    }
    else
    {
        fabric->ctx = NULL;
    }

    /* For non-CXI providers, clear FI_MR_BASIC (legacy flag).  For CXI, the
     * real fi_info from NULL-hints fi_getinfo already has the correct
     * mr_mode/max_msg_size/tx_attr/rx_attr — do NOT override them.        */
    if (!info->fabric_attr->prov_name ||
        strcmp(info->fabric_attr->prov_name, "cxi") != 0)
    {
        info->domain_attr->mr_mode = 0;
    }
#ifdef SST_HAVE_CRAY_DRC
    if (strstr(info->fabric_attr->prov_name, "gni") && fabric->auth_key)
    {
        info->domain_attr->auth_key = (uint8_t *)fabric->auth_key;
        info->domain_attr->auth_key_size = sizeof(struct fi_gni_raw_auth_key);
    }
#endif /* SST_HAVE_CRAY_DRC */

    result = fi_fabric(info->fabric_attr, &fabric->fabric, fabric->ctx);
    if (result != FI_SUCCESS)
    {
        fprintf(
            stderr,
            "opening fabric access failed with %d (%s). This is fatal.\n",
            result,
            fi_strerror(result));
        return;
    }
    result = fi_domain(fabric->fabric, info, &fabric->domain, fabric->ctx);
    if (result != FI_SUCCESS)
    {
        fprintf(
            stderr,
            "accessing domain failed with %d (%s). This is fatal.\n",
            result,
            fi_strerror(result));
        fprintf(
            stderr,
            "SST RDMA Dataplane failure.  fi_domain() has failed, which may "
            "mean that libfabric is defaulting to the wrong interface.  Check "
            "your FABRIC_IFACE environment variable (or specify one).\n");
        return;
    }

    /* CRITICAL: the fi_info returned by fi_getinfo has tx_attr->op_flags /
     * rx_attr->op_flags with FI_INJECT set by default for CXI.  libfabric's
     * simple (non-msg) calls like fi_read()/fi_write() implicitly use the
     * endpoint's default op_flags, so leaving FI_INJECT set causes CXI to
     * treat every fi_read/fi_write as an inject operation, capping transfer
     * size at tx_attr->inject_size (192 bytes on this system) and failing
     * larger transfers with EMSGSIZE ("Message too long").  Clear op_flags
     * before fi_endpoint() so normal-sized RMA reads/writes work.          */
    info->tx_attr->op_flags = 0;
    info->rx_attr->op_flags = 0;
    /* Do NOT override ep_attr->type here — the fi_info already has the
     * correct type from fi_getinfo and changing it after fi_domain causes
     * fi_endpoint to fail with EINVAL (CXI). */

    result = fi_endpoint(fabric->domain, info, &fabric->signal, fabric->ctx);
    if (result != FI_SUCCESS || !fabric->signal)
    {
        fprintf(
            stderr,
            "opening endpoint failed with %d (%s). This is fatal.\n",
            result,
            fi_strerror(result));
        return;
    }

    /* Query real max_msg_size from the created endpoint — necessary for CXI
     * where fi_getinfo with NULL hints returns max_msg_size=0.               */
    {
        size_t max_msg_size = 0;
        size_t optlen = sizeof(max_msg_size);
        if (fi_getopt(&fabric->signal->fid, FI_OPT_ENDPOINT,
                      FI_OPT_MAX_MSG_SIZE, &max_msg_size, &optlen) == FI_SUCCESS
            && max_msg_size > 0)
        {
            info->ep_attr->max_msg_size = max_msg_size;
            fprintf(stderr, "endpoint max_msg_size=%zu\n", max_msg_size);
        }
    }

    av_attr.type = FI_AV_MAP;
    av_attr.count = DP_AV_DEF_SIZE;
    av_attr.ep_per_node = 0;
    result = fi_av_open(fabric->domain, &av_attr, &fabric->av, fabric->ctx);
    if (result != FI_SUCCESS)
    {
        fprintf(
            stderr,
            "could not initialize address vector, failed with %d "
            "(%s). This is fatal.\n",
            result,
            fi_strerror(result));
        return;
    }
    result = fi_ep_bind(fabric->signal, &fabric->av->fid, 0);
    if (result != FI_SUCCESS)
    {
        fprintf(
            stderr,
            "could not bind endpoint to address vector, failed with "
            "%d (%s). This is fatal.\n",
            result,
            fi_strerror(result));
        return;
    }

    cq_attr.size = 0;
    cq_attr.format = FI_CQ_FORMAT_DATA;
    // (2025/09) segfault when using providers other than sockets
    // cq_attr.wait_obj = FI_WAIT_UNSPEC;
    // cq_attr.wait_cond = FI_CQ_COND_NONE;
    result =
        fi_cq_open(fabric->domain, &cq_attr, &fabric->cq_signal, fabric->ctx);
    if (result != FI_SUCCESS)
    {
        fprintf(
            stderr,
            "opening completion queue failed with %d (%s). This is fatal.\n",
            result,
            fi_strerror(result));
        return;
    }

    result = fi_ep_bind(
        fabric->signal, &fabric->cq_signal->fid, FI_TRANSMIT | FI_RECV);
    if (result != FI_SUCCESS)
    {
        fprintf(
            stderr,
            "could not bind endpoint to completion queue, failed "
            "with %d (%s). This is fatal.\n",
            result,
            fi_strerror(result));
        return;
    }

    result = fi_enable(fabric->signal);
    if (result != FI_SUCCESS)
    {
        fprintf(
            stderr,
            "enable endpoint, failed with %d (%s). This is fatal.\n",
            result,
            fi_strerror(result));
        return;
    }

    if (originfo) fi_freeinfo(originfo);
}

int handshake(struct fabric_state *fabric_state, MPI_Comm comm)
{
    char address[DP_AV_DEF_SIZE];
    size_t address_len = DP_AV_DEF_SIZE;
    int world_size = fabric_state->world_size;
    int rank = fabric_state->rank;

    int mr_rc = fi_mr_reg(
        fabric_state->domain,
        fabric_state->send_data,
        fabric_state->send_data_len,
        FI_WRITE | FI_REMOTE_READ,
        0,
        0,
        0,
        &fabric_state->mr,
        NULL);
    if (mr_rc != FI_SUCCESS)
    {
        fprintf(stderr, "fi_mr_reg failed: %s\n", fi_strerror(mr_rc));
        return 1;
    }

    /* CXI (FI_MR_ENDPOINT): bind MR to endpoint and enable it before use.
     * The provider-assigned key is only valid after fi_mr_enable().
     * On Perlmutter, fi_getinfo with NULL hints returns mr_mode=0 for CXI,
     * so is_mr_endpoint() detects CXI by provider name.                     */
    if (is_mr_endpoint(fabric_state))
    {
        int rc = fi_mr_bind(fabric_state->mr, &fabric_state->signal->fid, 0);
        if (rc != FI_SUCCESS)
        {
            fprintf(stderr, "fi_mr_bind (send) failed: %s\n", fi_strerror(rc));
            return 1;
        }
        rc = fi_mr_enable(fabric_state->mr);
        if (rc != FI_SUCCESS)
        {
            fprintf(stderr, "fi_mr_enable (send) failed: %s\n", fi_strerror(rc));
            return 1;
        }
    }
    fabric_state->key = fi_mr_key(fabric_state->mr);

    int status = fi_getname((fid_t)fabric_state->signal, address, &address_len);
    if (status != FI_SUCCESS)
    {
        fprintf(stderr, "fi_getname failed: %s\n", fi_strerror(status));
        return 1;
    }

    fabric_state->comm_partner   = (fi_addr_t *)malloc(world_size * sizeof(fi_addr_t));
    fabric_state->remote_key     = (uint64_t  *)malloc(world_size * sizeof(uint64_t));
    fabric_state->remote_address = (uint64_t  *)malloc(world_size * sizeof(uint64_t));

    char *address_data = (char *)malloc(world_size * address_len);
    for (int i = 0; i < address_len; i++)
    {
        address_data[rank * address_len + i] = address[i];
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, address_data, address_len, MPI_CHAR, comm);
    for (int i = 0; i < world_size; i++)
    {
        fi_av_insert(fabric_state->av, address_data + address_len * i, 1, &(fabric_state->comm_partner[i]), 0, NULL);
    }

    uint64_t *key_data = (uint64_t *)malloc(world_size * sizeof(uint64_t));
    key_data[rank] = fabric_state->key;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, key_data, 1, MPI_UINT64_T, comm);
    for (int i = 0; i < world_size; i++)
    {
        fabric_state->remote_key[i] = key_data[i];
    }

    size_t *pointer_addr_data = (size_t *)malloc(world_size * sizeof(size_t));
    /* With FI_MR_VIRT_ADDR the remote offset in fi_read is the virtual
     * address; without it (e.g. CXI with FI_MR_PROV_KEY) it is 0-based
     * from the MR registration base, so exchange 0.                         */
    pointer_addr_data[rank] = is_virt_addr(fabric_state)
                                  ? (size_t)fabric_state->send_data
                                  : 0;

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pointer_addr_data, 1, MPI_UNSIGNED_LONG, comm);
    for (int i = 0; i < world_size; i++)
    {
        fabric_state->remote_address[i] = pointer_addr_data[i];
    }

    free(address_data);
    free(key_data);
    free(pointer_addr_data);
    return 0;
}

int read_from_remote(struct fabric_state *fabric_state, int src, uint64_t offset)
{
    // register dest buffer; close previous recv MR first to avoid leaking it
    if (fabric_state->recv_mr)
        fi_close(&fabric_state->recv_mr->fid);
    fi_mr_reg(
        fabric_state->domain,
        fabric_state->recv_data,
        fabric_state->recv_data_len,
        FI_READ,
        0,
        0,
        0,
        &fabric_state->recv_mr,
        NULL);

    /* CXI (FI_MR_ENDPOINT): bind and enable recv MR before use. */
    if (is_mr_endpoint(fabric_state))
    {
        fi_mr_bind(fabric_state->recv_mr, &fabric_state->signal->fid, 0);
        fi_mr_enable(fabric_state->recv_mr);
    }

    void *memory_descriptor = NULL;
    if (is_local_mr_req(fabric_state))
    {
        memory_descriptor = fi_mr_desc(fabric_state->recv_mr);
    }

    size_t rc;
    do
    {
        rc = fi_read(
            fabric_state->signal,
            fabric_state->recv_data,
            fabric_state->recv_data_len,
            memory_descriptor,
            fabric_state->comm_partner[src],
            fabric_state->remote_address[src] + offset,
            fabric_state->remote_key[src],
            NULL);
    } while (rc == -EAGAIN);
    if (rc != 0)
    {
        fprintf(stderr, "fi_read failed with code %zd (%s).\n", (ssize_t)rc, fi_strerror((int)(ssize_t)rc));
        return (rc);
    }

    // (2025/09) segfault when using providers other than sockets
    // struct fi_cq_data_entry CQEntry = {0};
    // rc = fi_cq_sread(fabric_state->cq_signal, &CQEntry, 1, NULL, -1);
    // if (rc < 1)
    // {
    //     fprintf(stderr, "Received no completion event for remote read\n");
    //     return 1;
    // }

    for (;;)
    {
        struct fi_cq_data_entry CQEntry = {0};
        rc = fi_cq_read(fabric_state->cq_signal, &CQEntry, 1);
        if (rc == 1)
            break;
        if (rc == -FI_EAVAIL)
        {
            struct fi_cq_err_entry ee = {0};
            fi_cq_readerr(fabric_state->cq_signal, &ee, 0);
            /* prov_errno is provider-specific; fi_strerror() is only valid
             * for generic fi_errno values.  Use fi_cq_strerror() to get the
             * correct provider-aware error string.                          */
            char errbuf[256];
            const char *errstr = fi_cq_strerror(fabric_state->cq_signal,
                                                 ee.prov_errno, ee.err_data,
                                                 errbuf, sizeof(errbuf));
            fprintf(stderr,
                    "fi_cq_read failed: err=%d (%s) prov_errno=%d (%s)\n",
                    ee.err, fi_strerror(ee.err), ee.prov_errno,
                    errstr ? errstr : "(unknown)");
            return 1;
        }
    }

    return 0;
}

/* =========================================================================
 * Method 2: file-based handshake helpers
 * =========================================================================
 *
 * File naming convention:
 *   {dir}/{varname}.bin   — one shared file per variable; core rank i's
 *   CoreRecord lives at byte offset i * sizeof(struct CoreRecord).  Each
 *   rank pwrite()s only its own slot, so concurrent writers never touch
 *   each other's bytes; readers poll each slot's `ready` field (see
 *   CoreRecord in common.h) instead of the old per-rank file's existence.
 *
 * Directory resolution priority:
 *   1. user_dir argument (non-empty string)
 *   2. DDSTORE_HANDSHAKE_DIR environment variable
 *   3. "./ddstore_hs" (current working directory fallback)
 *
 * The resolved directory must be on a shared filesystem (e.g. Lustre)
 * visible to all nodes.  It is created automatically if it does not exist.
 *
 * Default poll timeout: DDSTORE_HANDSHAKE_TIMEOUT_S env var, default 300 s.
 * Poll interval: 50 ms.
 * ========================================================================= */

/* Resolve the handshake directory.
 * Returns a pointer to a static buffer — copy before next call.             */
const char *resolve_handshake_dir(const char *user_dir)
{
    static char resolved[4096];

    if (user_dir && user_dir[0] != '\0')
    {
        snprintf(resolved, sizeof(resolved), "%s", user_dir);
    }
    else
    {
        const char *env = getenv("DDSTORE_HANDSHAKE_DIR");
        if (env && env[0] != '\0')
            snprintf(resolved, sizeof(resolved), "%s", env);
        else
            snprintf(resolved, sizeof(resolved), "./ddstore_hs");
    }

    /* Create the directory if it does not exist (best-effort; ignore EEXIST). */
    mkdir(resolved, 0755);

    return resolved;
}

/* Build the canonical path for a variable's shared record file into `buf`. */
static void record_path(char *buf, size_t bufsz,
                        const char *dir, const char *varname)
{
    snprintf(buf, bufsz, "%s/%s.bin", dir, varname);
}

/* Return the configured timeout in seconds (default 300).                   */
static int handshake_timeout_s(void)
{
    const char *env = getenv("DDSTORE_HANDSHAKE_TIMEOUT_S");
    if (env) return atoi(env);
    return 300;
}

/* --------------------------------------------------------------------------
 * handshake_write()
 *
 * Called by each core rank after init_fabric() and fi_mr_reg().
 * Writes this rank's CoreRecord into its slot of the shared
 * {resolved_dir}/{varname}.bin file, at byte offset rank * sizeof(CoreRecord).
 * The directory is resolved via resolve_handshake_dir().
 * -------------------------------------------------------------------------- */
int handshake_write(struct fabric_state *fs,
                    const char *dir, const char *varname, int rank,
                    long nrows, int disp, int itemsize)
{
    const char *rdir = resolve_handshake_dir(dir);

    /* Build the CoreRecord for this rank. `ready` is left 0 here and
     * committed separately below, after the rest of the slot is durable. */
    struct CoreRecord rec;
    memset(&rec, 0, sizeof(rec));

    rec.fabric_address_len = DP_AV_DEF_SIZE;
    int status = fi_getname((fid_t)fs->signal,
                            rec.fabric_address, &rec.fabric_address_len);
    if (status != FI_SUCCESS)
    {
        fprintf(stderr, "[handshake_write] fi_getname failed: %s\n",
                fi_strerror(status));
        return 1;
    }

    rec.key          = fs->key;
    /* With FI_MR_VIRT_ADDR the remote offset in fi_read is the virtual
     * address; without it (e.g. CXI with FI_MR_PROV_KEY) it is 0-based
     * from the MR registration base — same rule as method=1's handshake(). */
    rec.base_address = is_virt_addr(fs) ? (uint64_t)(uintptr_t)fs->send_data : 0;
    rec.nrows        = nrows;
    rec.disp         = disp;
    rec.itemsize     = itemsize;
    rec.ready        = 0;

    char bin_path[4096];
    record_path(bin_path, sizeof(bin_path), rdir, varname);

    /* O_CREAT without O_TRUNC: this file is shared by every core rank, each
     * writing only its own slot, so no writer may ever truncate/recreate it
     * (that would clobber slots other ranks already committed).             */
    int fd = open(bin_path, O_CREAT | O_WRONLY, 0644);
    if (fd < 0)
    {
        fprintf(stderr, "[handshake_write] cannot open %s: ", bin_path);
        perror("");
        return 1;
    }

    off_t slot_off = (off_t)rank * (off_t)sizeof(struct CoreRecord);

    /* Phase 1: write the record body (ready=0) and fsync it durable. */
    if (pwrite(fd, &rec, sizeof(rec), slot_off) != (ssize_t)sizeof(rec) ||
        fsync(fd) != 0)
    {
        fprintf(stderr, "[handshake_write] write/fsync failed for %s: ", bin_path);
        perror("");
        close(fd);
        return 1;
    }

    /* Phase 2: commit readiness with its own write+fsync, after the body
     * above is durable, so a reader that observes CORE_RECORD_READY_MAGIC
     * never sees a partially-written record.                                */
    uint32_t ready_magic = CORE_RECORD_READY_MAGIC;
    off_t ready_off = slot_off + (off_t)offsetof(struct CoreRecord, ready);
    if (pwrite(fd, &ready_magic, sizeof(ready_magic), ready_off) != (ssize_t)sizeof(ready_magic) ||
        fsync(fd) != 0)
    {
        fprintf(stderr, "[handshake_write] ready commit failed for %s: ", bin_path);
        perror("");
        close(fd);
        return 1;
    }

    close(fd);

    fprintf(stderr, "[handshake_write] rank %d wrote slot in %s\n", rank, bin_path);
    return 0;
}

/* --------------------------------------------------------------------------
 * handshake_read()
 *
 * Called by each core rank after handshake_write().
 * Polls the shared {resolved_dir}/{varname}.bin file until all n_core slots
 * report ready (see CoreRecord.ready in common.h), then reads them and
 * populates:
 *   fs->comm_partner[0..n_core-1]   (fi_addr_t, via fi_av_insert)
 *   fs->remote_key[0..n_core-1]
 *   fs->remote_address[0..n_core-1]
 *   lenlist[0..n_core-1]            (raw nrows, NOT prefix-summed)
 *   *out_disp, *out_itemsize        (from rank-0 record; assumed uniform)
 *
 * Returns 0 on success, non-zero on error or timeout.
 * -------------------------------------------------------------------------- */
int handshake_read(struct fabric_state *fs,
                   const char *dir, const char *varname,
                   int n_core, int my_rank,
                   long *lenlist, int *out_disp, int *out_itemsize)
{
    const char *rdir = resolve_handshake_dir(dir);
    int timeout_s = handshake_timeout_s();
    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    /* Allocate arrays sized for n_core peers. */
    fs->comm_partner   = (fi_addr_t *)malloc(n_core * sizeof(fi_addr_t));
    fs->remote_key     = (uint64_t  *)malloc(n_core * sizeof(uint64_t));
    fs->remote_address = (uint64_t  *)malloc(n_core * sizeof(uint64_t));
    if (!fs->comm_partner || !fs->remote_key || !fs->remote_address)
    {
        fprintf(stderr, "[handshake_read] malloc failed\n");
        return 1;
    }

    char path[4096];
    record_path(path, sizeof(path), rdir, varname);

    /* Poll until every slot's `ready` field reads back CORE_RECORD_READY_MAGIC.
     * A slot that was never written (sparse hole, or the file doesn't exist
     * yet) reads back as zero, which never matches the magic.               */
    for (;;)
    {
        int ready = 0;
        int fd = open(path, O_RDONLY);
        if (fd >= 0)
        {
            for (int i = 0; i < n_core; i++)
            {
                uint32_t flag = 0;
                off_t off = (off_t)i * (off_t)sizeof(struct CoreRecord) +
                            (off_t)offsetof(struct CoreRecord, ready);
                if (pread(fd, &flag, sizeof(flag), off) == (ssize_t)sizeof(flag) &&
                    flag == CORE_RECORD_READY_MAGIC)
                    ready++;
            }
            close(fd);
        }
        if (ready == n_core)
            break;

        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        double elapsed = (ts_now.tv_sec  - ts_start.tv_sec) +
                         (ts_now.tv_nsec - ts_start.tv_nsec) * 1e-9;
        if (elapsed > timeout_s)
        {
            fprintf(stderr,
                    "[handshake_read] timeout after %.0f s waiting for "
                    "%d/%d core records (var=%s, dir=%s)\n",
                    elapsed, ready, n_core, varname, rdir);
            return 1;
        }
        usleep(50000); /* 50 ms */
    }

    /* Read all records. */
    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "[handshake_read] cannot open %s: ", path);
        perror("");
        return 1;
    }

    for (int i = 0; i < n_core; i++)
    {
        struct CoreRecord rec;
        off_t off = (off_t)i * (off_t)sizeof(struct CoreRecord);
        if (pread(fd, &rec, sizeof(rec), off) != (ssize_t)sizeof(rec))
        {
            fprintf(stderr, "[handshake_read] read failed for slot %d of %s\n", i, path);
            close(fd);
            return 1;
        }

        /* Insert fabric address into the address vector. */
        int rc = fi_av_insert(fs->av,
                              rec.fabric_address, 1,
                              &fs->comm_partner[i], 0, NULL);
        if (rc != 1)
        {
            fprintf(stderr,
                    "[handshake_read] fi_av_insert failed for rank %d (rc=%d)\n",
                    i, rc);
            close(fd);
            return 1;
        }

        fs->remote_key[i]     = rec.key;
        fs->remote_address[i] = rec.base_address;
        lenlist[i]            = rec.nrows;

        if (i == 0)
        {
            if (out_disp)     *out_disp     = rec.disp;
            if (out_itemsize) *out_itemsize = rec.itemsize;
        }
    }
    close(fd);

    fs->world_size = n_core;
    (void)my_rank;
    return 0;
}

/* --------------------------------------------------------------------------
 * handshake_join()
 *
 * Called by extra members (no MPI, no data to publish).
 * Identical to handshake_read() except it never writes anything.
 * Blocks until all n_core record files are present (or timeout).
 * -------------------------------------------------------------------------- */
int handshake_join(struct fabric_state *fs,
                   const char *dir, const char *varname,
                   int n_core,
                   long *lenlist, int *out_disp, int *out_itemsize)
{
    return handshake_read(fs, dir, varname, n_core, -1,
                          lenlist, out_disp, out_itemsize);
}