#pragma once

#include "rdma/fabric.h"
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <mpi.h>

#define DP_AV_DEF_SIZE 512
#define COMM_FILE_WRITER_TO_READER "./writer_address.bin"
#define MAX_WORLD_SIZE 8192

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

        fi_addr_t comm_partner[MAX_WORLD_SIZE];
        char *send_data;
        size_t send_data_len;
        char *recv_data;
        size_t recv_data_len;
        struct fid_mr *mr;
        uint64_t key;
        uint64_t remote_key[MAX_WORLD_SIZE];
        uint64_t remote_address[MAX_WORLD_SIZE];

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

#ifdef __cplusplus
}
#endif