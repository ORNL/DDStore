from mpi4py import MPI
import numpy as np
import os

import torch
from torch.utils.data import Dataset

import pyddstore as dds


def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


class DistDataset(Dataset):
    """Distributed dataset class"""

    def __init__(self, data, label, comm=MPI.COMM_WORLD, ddstore_width=None,
                 device=None, add_device=None):
        super().__init__()

        self.dataset = list()
        self.label = label
        self.comm = comm
        # None -> get() allocates host buffers (default, unchanged).
        # torch.device/"cuda"/"cuda:N" -> get() allocates its destination
        # buffer directly on that device (GPUDirect RDMA, Phase 1); requires
        # DDSTORE_METHOD in (1, 2) and DDSTORE_FABRIC=cxi (pyddstore raises
        # a clear error otherwise).
        self.device = device
        # None -> add() makes a private host copy of this rank's shard
        # (default, unchanged). torch.device/"cuda"/"cuda:N" -> the shard is
        # stacked directly on that device and add() registers it in place,
        # no host copy (GPUDirect RDMA, Phase 2) -- same DDSTORE_METHOD/
        # DDSTORE_FABRIC requirements as `device` above. Independent of
        # `device`: this controls the SOURCE side, `device` controls the
        # DESTINATION side, so source and destination can be tested
        # separately or together.
        self.add_device = add_device
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        print("init", self.rank, self.comm_size)
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        self.ddstore_comm = self.comm.Split(self.rank // self.ddstore_width, self.rank)
        self.ddstore_comm_rank = self.ddstore_comm.Get_rank()
        self.ddstore_comm_size = self.ddstore_comm.Get_size()

        ddstore_method = int(os.getenv("DDSTORE_METHOD", "0"))
        print("DDStore method:", ddstore_method)
        handshake_dir = os.getenv("DDSTORE_HANDSHAKE_DIR", "./ddstore_hs")

        if ddstore_method == 2 and self.ddstore_width != self.comm_size:
            # File-based handshake: each Split group would publish into the
            # same shared {varname}.bin file, so more than one group sharing
            # a handshake_dir would silently collide.
            raise NotImplementedError(
                "method=2 does not yet support ddstore_width < comm_size "
                "(multiple core groups would collide on the same "
                "handshake_dir)"
            )

        self.ddstore = dds.PyDDStore(
            self.ddstore_comm, method=ddstore_method, handshake_dir=handshake_dir
        )
        print("FABRIC_IFACE:", os.environ.get("FABRIC_IFACE", "n/a (method=0)"))

        ## set total before set subset
        self.total_ns = len(data)
        print("init", self.total_ns)

        # WHEN READY FOR WHOLE DATA SET CHANGE THE RANGE TO range(len(data))
        rx = list(nsplit(range(len(data)), self.ddstore_comm_size))[
            self.ddstore_comm_rank
        ]

        for i in rx:
            self.dataset.append(data[i])

        print(self.rank, len(self.dataset))

        # Label stays host-only regardless of add_device -- see get()'s
        # matching comment; a GPU-resident int32 label buffer buys nothing.
        self.labels = [label for _, label in self.dataset]
        self.labels = np.array(self.labels, dtype=np.int32)
        self.labels = np.ascontiguousarray(self.labels)

        if self.add_device is not None:
            # GPUDirect RDMA source (Phase 2): stack directly on device, no
            # host round-trip. torch.stack (not cat) keeps one row per image
            # (nrows, 784) -- see the np.stack comment below for why that
            # shape matters to ddstore.add()'s disp inference.
            self.data = torch.stack(
                [d.reshape(-1) for d, _ in self.dataset]
            ).to(self.add_device)
            self.data = self.data.contiguous()
        else:
            data_list = list()
            for data, _ in self.dataset:
                val = data.cpu().numpy()
                val = val.flatten()
                data_list.append(val)

            # np.stack (not concatenate) keeps one row per image (nrows, 784)
            # so ddstore.add() infers disp=784 instead of flattening into a
            # single (nrows*784,) vector, which it would read back as disp=1.
            self.data = np.stack(data_list)
            self.data = np.ascontiguousarray(self.data)

        self.ddstore.add(f"{self.label}data", self.data)
        self.ddstore.add(f"{self.label}labels", self.labels)

    def len(self):
        return self.total_ns

    def __len__(self):
        return self.len()

    def get(self, idx, device=None):
        ## first dim must be the row count (1), not the flattened feature
        ## width, since ddstore.get() infers count from arr.shape[0]
        # Label stays host-only regardless of `device` -- both training
        # loops discard it, so a GPU-resident label buffer would add
        # complexity for no benefit.
        label = np.zeros(1, dtype=np.int32)
        if device is not None:
            # RDMA fully overwrites this buffer, so torch.empty (not
            # zeros): for MNIST's mostly-zero background pixels, an
            # all-zeros buffer would look deceptively plausible even if the
            # read silently no-op'd.
            val = torch.empty((1, 28 * 28), dtype=torch.float32, device=device)
        else:
            val = np.zeros((1, 28 * 28), dtype=np.float32)
            val = np.ascontiguousarray(val)
            assert val.data.contiguous
        self.ddstore.get(f"{self.label}data", val, idx)
        self.ddstore.get(f"{self.label}labels", label, idx)
        # print("rank", self.rank, "fetching idx", idx)
        if device is None:
            val = torch.tensor(val)
        val = torch.reshape(val, (1, 28, 28))
        return (val, label[0])

    def __getitem__(self, idx):
        return self.get(idx, device=self.device)


class DistDatasetReader(Dataset):
    """Distributed dataset class — extra (read-only) member.

    Joins a variable published by a core group (see DistDataset) via
    DDStore method=2's file-based handshake. Owns no MPI communicator and no
    local copy of the data — every __getitem__ is an RDMA read against a
    core rank's memory.
    """

    def __init__(self, label, handshake_dir, n_core, device=None):
        super().__init__()
        self.label = label
        # See DistDataset.__init__ for what `device` does.
        self.device = device

        self.ddstore = dds.PyDDStore(
            None, method=2, handshake_dir=handshake_dir, n_core=n_core
        )
        print("FABRIC_IFACE:", os.environ.get("FABRIC_IFACE", "n/a"))
        self.ddstore.join(f"{label}data")
        self.ddstore.join(f"{label}labels")

        self.total_ns, self.data_disp, self.data_itemsize = self.ddstore.info(
            f"{label}data"
        )
        self.side = int(round(self.data_disp**0.5))
        if self.side * self.side != self.data_disp:
            raise ValueError(
                f"joined '{label}data' has disp={self.data_disp}, "
                "which is not a perfect square (expected a flattened square image)"
            )

    def len(self):
        return self.total_ns

    def __len__(self):
        return self.len()

    def get(self, idx, device=None):
        ## first dim must be the row count (1), not the flattened feature
        ## width, since ddstore.get() infers count from arr.shape[0]
        # Label stays host-only regardless of `device` -- see DistDataset.get().
        label = np.zeros(1, dtype=np.int32)
        if device is not None:
            # torch.empty, not zeros -- see DistDataset.get() for why.
            val = torch.empty((1, self.data_disp), dtype=torch.float32, device=device)
        else:
            val = np.zeros((1, self.data_disp), dtype=np.float32)
            val = np.ascontiguousarray(val)
            assert val.data.contiguous
        self.ddstore.get(f"{self.label}data", val, idx)
        self.ddstore.get(f"{self.label}labels", label, idx)
        if device is None:
            val = torch.tensor(val)
        val = torch.reshape(val, (1, self.side, self.side))
        return (val, label[0])

    def __getitem__(self, idx):
        return self.get(idx, device=self.device)
