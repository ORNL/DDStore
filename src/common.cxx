#include "common.h"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

void init_fabric(struct fabric_state *fabric)
{
    struct fi_info *hints, *info, *originfo, *useinfo;
    struct fi_av_attr av_attr = {FI_AV_UNSPEC};
    struct fi_cq_attr cq_attr = {0};
    char *ifname;
    int result;

    hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_SEND | FI_RECV | FI_REMOTE_READ |
                  FI_REMOTE_WRITE | FI_RMA | FI_READ | FI_WRITE;
    hints->mode = FI_CONTEXT | FI_LOCAL_MR | FI_CONTEXT2 | FI_MSG_PREFIX |
                  FI_ASYNC_IOV | FI_RX_CQ_DATA;
    hints->domain_attr->mr_mode = FI_MR_BASIC;
    hints->domain_attr->control_progress = FI_PROGRESS_AUTO;
    hints->domain_attr->data_progress = FI_PROGRESS_AUTO;
    hints->domain_attr->threading = FI_THREAD_DOMAIN;
    hints->ep_attr->type = FI_EP_RDM;

    ifname = getenv("FABRIC_IFACE");

    fabric->info = NULL;

    int version = fi_version();
    fi_getinfo(version, NULL, NULL, 0, hints, &info);
    if (!info)
    {
        fprintf(stderr, "no fabrics detected.\n");
        fabric->info = NULL;
        return;
    }
    fi_freeinfo(hints);

    originfo = info;
    useinfo = NULL;
    while (info)
    {
        char *prov_name = info->fabric_attr->prov_name;
        char *domain_name = info->domain_attr->name;

        // (2025/09) need to hardcode to avoid conflict with MPI
        if (ifname && (strcmp(ifname, domain_name) == 0) && (strcmp(prov_name, "tcp;ofi_rxm") == 0))
        {
            fprintf(stderr, "using interface set by FABRIC_IFACE.\n");
            useinfo = info;
            break;
        }
        if ((((strcmp(prov_name, "verbs") == 0) && info->src_addr) ||
             (strcmp(prov_name, "gni") == 0) ||
             (strcmp(prov_name, "psm2") == 0)) &&
            (!useinfo || !ifname ||
             (strcmp(useinfo->domain_attr->name, ifname) != 0)))
        {
            fprintf(
                stderr,
                "seeing candidate fabric %s, will use this unless we "
                "see something better.\n",
                prov_name);
            useinfo = info;
        }
        else if (
            ((strstr(prov_name, "verbs") && info->src_addr) ||
             strstr(prov_name, "gni") || strstr(prov_name, "psm2")) &&
            !useinfo)
        {
            fprintf(
                stderr,
                "seeing candidate fabric %s, will use this unless we "
                "see something better.\n",
                prov_name);
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

    info->domain_attr->mr_mode = FI_MR_BASIC;
#ifdef SST_HAVE_CRAY_DRC
    if (strstr(info->fabric_attr->prov_name, "gni") && fabric->auth_key)
    {
        info->domain_attr->auth_key = (uint8_t *)fabric->auth_key;
        info->domain_attr->auth_key_size = sizeof(struct fi_gni_raw_auth_key);
    }
#endif /* SST_HAVE_CRAY_DRC */
    fabric->info = fi_dupinfo(info);
    if (!fabric->info)
    {
        fprintf(stderr, "copying the fabric info failed.\n");
        return;
    }

    // fprintf(
    //     stderr,
    //     "Fabric parameters to use at fabric initialization: %s\n",
    //     fi_tostr(fabric->info, FI_TYPE_INFO));

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
    info->ep_attr->type = FI_EP_RDM;
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

    fi_freeinfo(originfo);
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
    pointer_addr_data[rank] = (size_t)fabric_state->send_data;

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
    void *memory_descriptor = NULL;
    if (is_local_mr_req(fabric_state))
    {
        memory_descriptor = fi_mr_desc(fabric_state->recv_mr);
    }

    size_t rc;
    // fprintf(stderr, "fabric_state->remote_address: %llu\n", fabric_state->remote_address[src]);
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
        fprintf(stderr, "fi_read failed with code %zu.\n", rc);
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
            fprintf(stderr, "fi_cq_read failed with error: prov_errno=%d (%s)\n",
                    ee.prov_errno, fi_strerror(ee.prov_errno));
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
 *   {dir}/{varname}.bin   — one combined file per variable, holding all
 *                           n_core CoreRecords, written once by core rank 0
 *                           after an MPI_Allgather among core ranks.
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

/* Build the canonical path for a variable's combined record file into `buf`. */
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
 * Exchanges CoreRecords with all other core ranks via MPI_Allgather over
 * `comm` — no filesystem round-trip needed, since core ranks already share
 * a communicator.  Builds this rank's address vector / remote key / remote
 * address arrays from the gathered records.  Rank 0 additionally publishes
 * the combined array to {resolved_dir}/{varname}.bin (tmp + fsync + rename)
 * so extra members can join later.
 * -------------------------------------------------------------------------- */
int handshake_write(struct fabric_state *fs, MPI_Comm comm,
                    const char *dir, const char *varname,
                    int n_core, long nrows, int disp, int itemsize,
                    long *lenlist)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    /* Build this rank's CoreRecord. */
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
    rec.base_address = (uint64_t)(uintptr_t)fs->send_data;
    rec.nrows        = nrows;
    rec.disp         = disp;
    rec.itemsize     = itemsize;

    /* Exchange records among core ranks directly over MPI. */
    std::vector<struct CoreRecord> all_recs(n_core);
    MPI_Allgather(&rec, sizeof(rec), MPI_BYTE,
                 all_recs.data(), sizeof(rec), MPI_BYTE, comm);

    /* Build this rank's address vector / remote key / remote address
     * arrays from the gathered records. */
    fs->comm_partner   = (fi_addr_t *)malloc(n_core * sizeof(fi_addr_t));
    fs->remote_key     = (uint64_t  *)malloc(n_core * sizeof(uint64_t));
    fs->remote_address = (uint64_t  *)malloc(n_core * sizeof(uint64_t));
    if (!fs->comm_partner || !fs->remote_key || !fs->remote_address)
    {
        fprintf(stderr, "[handshake_write] malloc failed\n");
        return 1;
    }
    for (int i = 0; i < n_core; i++)
    {
        int rc = fi_av_insert(fs->av, all_recs[i].fabric_address, 1,
                              &fs->comm_partner[i], 0, NULL);
        if (rc != 1)
        {
            fprintf(stderr,
                    "[handshake_write] fi_av_insert failed for rank %d (rc=%d)\n",
                    i, rc);
            return 1;
        }
        fs->remote_key[i]     = all_recs[i].key;
        fs->remote_address[i] = all_recs[i].base_address;
        lenlist[i]            = all_recs[i].nrows;
    }
    fs->world_size = n_core;

    /* Rank 0 publishes the combined record array for extra members.  Write
     * to a temp file first, fsync, then atomically rename into place so
     * readers polling the final path never observe a partially-written or
     * truncated-then-being-rewritten record.                                */
    if (rank == 0)
    {
        const char *rdir = resolve_handshake_dir(dir);
        char bin_path[4096];
        char tmp_path[4096 + 32];
        record_path(bin_path, sizeof(bin_path), rdir, varname);
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp.%d", bin_path, (int)getpid());

        FILE *f = fopen(tmp_path, "wb");
        if (!f)
        {
            fprintf(stderr, "[handshake_write] cannot open %s: ", tmp_path);
            perror("");
            return 1;
        }
        if (fwrite(all_recs.data(), sizeof(struct CoreRecord), n_core, f) != (size_t)n_core)
        {
            fprintf(stderr, "[handshake_write] fwrite failed for %s\n", tmp_path);
            fclose(f);
            unlink(tmp_path);
            return 1;
        }
        if (fflush(f) != 0 || fsync(fileno(f)) != 0)
        {
            fprintf(stderr, "[handshake_write] fsync failed for %s: ", tmp_path);
            perror("");
            fclose(f);
            unlink(tmp_path);
            return 1;
        }
        fclose(f);

        if (rename(tmp_path, bin_path) != 0)
        {
            fprintf(stderr, "[handshake_write] rename %s -> %s failed: ", tmp_path, bin_path);
            perror("");
            unlink(tmp_path);
            return 1;
        }

        fprintf(stderr, "[handshake_write] wrote %s (%d records)\n", bin_path, n_core);
    }

    return 0;
}

/* --------------------------------------------------------------------------
 * handshake_join()
 *
 * Called by extra members (no MPI communicator, no data to publish).
 * Polls for {resolved_dir}/{varname}.bin (the combined record file written
 * by core rank 0 in handshake_write()) to reach its expected size, reads
 * it, and populates:
 *   fs->comm_partner[0..n_core-1]   (fi_addr_t, via fi_av_insert)
 *   fs->remote_key[0..n_core-1]
 *   fs->remote_address[0..n_core-1]
 *   lenlist[0..n_core-1]            (raw nrows, NOT prefix-summed)
 *   *out_disp, *out_itemsize        (from record 0; assumed uniform)
 *
 * Returns 0 on success, non-zero on error or timeout.
 * -------------------------------------------------------------------------- */
int handshake_join(struct fabric_state *fs,
                   const char *dir, const char *varname,
                   int n_core,
                   long *lenlist, int *out_disp, int *out_itemsize)
{
    const char *rdir = resolve_handshake_dir(dir);
    int timeout_s = handshake_timeout_s();
    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    char path[4096];
    record_path(path, sizeof(path), rdir, varname);
    size_t expected_size = (size_t)n_core * sizeof(struct CoreRecord);

    /* Poll until the combined record file is fully written. */
    for (;;)
    {
        struct stat st;
        if (stat(path, &st) == 0 && st.st_size == (off_t)expected_size)
            break;

        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        double elapsed = (ts_now.tv_sec  - ts_start.tv_sec) +
                         (ts_now.tv_nsec - ts_start.tv_nsec) * 1e-9;
        if (elapsed > timeout_s)
        {
            fprintf(stderr,
                    "[handshake_join] timeout after %.0f s waiting for "
                    "%s (var=%s, dir=%s)\n",
                    elapsed, path, varname, rdir);
            return 1;
        }
        usleep(50000); /* 50 ms */
    }

    std::vector<struct CoreRecord> all_recs(n_core);
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "[handshake_join] cannot open %s: ", path);
        perror("");
        return 1;
    }
    if (fread(all_recs.data(), sizeof(struct CoreRecord), n_core, f) != (size_t)n_core)
    {
        fprintf(stderr, "[handshake_join] fread failed for %s\n", path);
        fclose(f);
        return 1;
    }
    fclose(f);

    fs->comm_partner   = (fi_addr_t *)malloc(n_core * sizeof(fi_addr_t));
    fs->remote_key     = (uint64_t  *)malloc(n_core * sizeof(uint64_t));
    fs->remote_address = (uint64_t  *)malloc(n_core * sizeof(uint64_t));
    if (!fs->comm_partner || !fs->remote_key || !fs->remote_address)
    {
        fprintf(stderr, "[handshake_join] malloc failed\n");
        return 1;
    }

    for (int i = 0; i < n_core; i++)
    {
        int rc = fi_av_insert(fs->av, all_recs[i].fabric_address, 1,
                              &fs->comm_partner[i], 0, NULL);
        if (rc != 1)
        {
            fprintf(stderr,
                    "[handshake_join] fi_av_insert failed for rank %d (rc=%d)\n",
                    i, rc);
            return 1;
        }
        fs->remote_key[i]     = all_recs[i].key;
        fs->remote_address[i] = all_recs[i].base_address;
        lenlist[i]            = all_recs[i].nrows;
    }

    if (out_disp)     *out_disp     = all_recs[0].disp;
    if (out_itemsize) *out_itemsize = all_recs[0].itemsize;

    fs->world_size = n_core;
    return 0;
}