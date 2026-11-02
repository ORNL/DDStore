#!/usr/bin/env python3
"""Report what a DDStore integration needs to know about this environment.

Read-only: imports pyddstore and torch, prints settings, starts no MPI or
fabric. Run on a login node or inside a job step.
"""

import os
import shutil
import sys


def show(key, value):
    print(f"{key:28s} {value}")


def main():
    env = dict(os.environ)  # before pyddstore sets FI_CXI_DISABLE_HOST_REGISTER
    show("python", sys.version.split()[0])
    try:
        import torch

        gpu = torch.cuda.is_available()
        show(
            "torch",
            f"{torch.__version__} (gpu: {gpu}, hip: {torch.version.hip or '-'})",
        )
        import torch.distributed as dist

        show(
            "torch.distributed",
            "initialized" if dist.is_initialized() else "not initialized",
        )
    except ImportError:
        show("torch", "not installed (pyddstore.torch unavailable)")

    try:
        import pyddstore
    except ImportError as e:
        show("pyddstore", f"not importable: {e}")
        return 1
    show(
        "pyddstore", f"{pyddstore.__version__} ({os.path.dirname(pyddstore.__file__)})"
    )
    try:
        have_mpi = pyddstore.HAVE_MPI
    except ImportError as e:
        show("pyddstore._core", f"not built: {e}")
        return 1
    show("HAVE_MPI", have_mpi)
    show("methods", "0, 1, 2" if have_mpi else "1, 2")
    if have_mpi:
        try:
            import mpi4py

            show("mpi4py", mpi4py.__version__)
        except ImportError:
            show("mpi4py", "missing (needed by an MPI build)")

    show("default method", os.environ.get("DDSTORE_METHOD", "0" if have_mpi else "1"))
    for var in (
        "DDSTORE_COMM",
        "DDSTORE_FABRIC",
        "DDSTORE_HANDSHAKE_DIR",
        "DDSTORE_NIC_MAP",
        "DDSTORE_FORK_WORKERS",
        "FI_MR_CACHE_MONITOR",
        "FI_CXI_DISABLE_HOST_REGISTER",
        "SLINGSHOT_VNIS",
    ):
        show(var, env.get(var, "(unset)"))

    if (
        env.get("FI_MR_CACHE_MONITOR") == "kdreg2"
        and env.get("DDSTORE_FORK_WORKERS") != "1"
    ):
        print(
            "note: kdreg2 fails in forked DataLoader workers; set DDSTORE_FORK_WORKERS=1 to use them"
        )
    show("slurm job", os.environ.get("SLURM_JOB_ID", "(not in a job)"))
    show(
        "system",
        os.environ.get("NERSC_HOST") or os.environ.get("LMOD_SYSTEM_NAME") or "unknown",
    )
    show(
        "cxi devices",
        (
            " ".join(sorted(os.listdir("/sys/class/cxi")))
            if os.path.isdir("/sys/class/cxi")
            else "none"
        ),
    )
    show("srun", shutil.which("srun") or "not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
