from mpi4py import MPI
import numpy as np

import torch
from torch.utils.data import Dataset

# from hydragnn.utils.abstractbasedataset import AbstractBaseDataset

try:
    import pyddstore as dds
except ImportError:
    pass

def nsplit(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

class DistDataset(Dataset):
    """Distributed dataset class"""

    def __init__(self, data, label, comm=MPI.COMM_WORLD, ddstore_width=None):
        super().__init__()
        
        self.dataset = list()
        self.label = label
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        print("init", self.rank, self.comm_size)
        self.ddstore_width = (
            ddstore_width if ddstore_width is not None else self.comm_size
        )
        self.ddstore_comm = self.comm.Split(self.rank // self.ddstore_width, self.rank)
        self.ddstore_comm_rank = self.ddstore_comm.Get_rank()
        self.ddstore_comm_size = self.ddstore_comm.Get_size()
        self.ddstore = dds.PyDDStore(self.ddstore_comm)

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
        self.data = list()
        self.labels = list()

        nbytes = 0
        for (data, label) in self.dataset:
            val = data.cpu().numpy()
            val = val.flatten()            
            self.data.append(val)
            self.labels.append(label)

        self.data = np.concatenate(self.data)
        self.data = np.ascontiguousarray(self.data)

        
        self.labels = np.array(self.labels, dtype=np.int32)
        self.labels = np.ascontiguousarray(self.labels)

        self.ddstore.add(f"{self.label}data", self.data)
        self.ddstore.add(f"{self.label}labels", self.labels)

    def len(self):
        return self.total_ns

    def __len__(self):
        return self.len()

    def get(self, idx):
        val = np.zeros(28 * 28, dtype=np.float32)
        label = np.zeros(1, dtype=np.int32)
        val = np.ascontiguousarray(val)
        assert val.data.contiguous
        self.ddstore.get(f"{self.label}data", val, idx)
        self.ddstore.get(f"{self.label}labels", label, idx)
        # print("rank", self.rank, "fetching idx", idx)
        val = torch.tensor(val)
        val = torch.reshape(val, (1, 28, 28))
        return (val, label[0])

    def __getitem__(self, idx):
        return self.get(idx)
