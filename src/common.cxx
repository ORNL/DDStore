#include "common.h"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    int src = (rank - 1) % fabric_state->world_size;

    fi_mr_reg(
        fabric_state->domain,
        fabric_state->send_data,
        fabric_state->send_data_len,
        FI_WRITE | FI_REMOTE_READ,
        0,
        0,
        0,
        &fabric_state->mr,
        NULL);
    fabric_state->key = fi_mr_key(fabric_state->mr);

    int status = fi_getname((fid_t)fabric_state->signal, address, &address_len);
    if (status != FI_SUCCESS)
    {
        fprintf(stderr, "fi_getname failed: %s\n", fi_strerror(status));
        return 1;
    }

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

    return 0;
}

int read_from_remote(struct fabric_state *fabric_state, int src, uint64_t offset)
{
    // register dest buffer
    fi_mr_reg(
        fabric_state->domain,
        fabric_state->recv_data,
        fabric_state->recv_data_len,
        FI_READ,
        0,
        0,
        0,
        &fabric_state->mr,
        NULL);
    void *memory_descriptor = NULL;
    if (is_local_mr_req(fabric_state))
    {
        memory_descriptor = fi_mr_desc(fabric_state->mr);
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