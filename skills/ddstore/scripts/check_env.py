#!/usr/bin/env python3
"""Report what a DDStore integration needs to know about this environment.

Read-only: imports pyddstore and torch, prints settings, starts no MPI or
fabric. Run on a login node or, better, inside a job step. Python >= 3.9.

  python check_env.py          # table
  python check_env.py --json   # one JSON object
"""

import json
import os
import shutil
import sys

FABRIC_VARS = (
    "DDSTORE_FABRIC",
    "DDSTORE_NIC_MAP",
    "DDSTORE_FORK_WORKERS",
    "FI_MR_CACHE_MONITOR",
    "FI_CXI_DISABLE_HOST_REGISTER",
    "SLINGSHOT_VNIS",
)


def collect():
    user_env = {v: os.environ.get(v) for v in FABRIC_VARS}  # before pyddstore
    info = {"python": sys.version.split()[0], "notes": []}
    torch_dist_init = False
    try:
        import torch
        import torch.distributed as dist

        torch_dist_init = dist.is_available() and dist.is_initialized()
        info["torch"] = torch.__version__
        info["gpu_runtime"] = (
            f"hip {torch.version.hip}"
            if torch.version.hip
            else f"cuda {torch.version.cuda}"
        )
        info["visible_gpus"] = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        info["torch_distributed"] = (
            "initialized" if torch_dist_init else "not initialized"
        )
    except ImportError:
        info["torch"] = None
        info["notes"].append("torch not installed: pyddstore.torch is unavailable")

    try:
        import pyddstore

        info["pyddstore"] = pyddstore.__version__
        info["pyddstore_path"] = os.path.dirname(pyddstore.__file__)
        have_mpi = pyddstore.HAVE_MPI  # loads the compiled extension
    except ImportError as e:
        info["pyddstore"] = None
        info["notes"].append(f"pyddstore not importable or not built: {e}")
        return info
    info["have_mpi"] = have_mpi
    info["methods_in_build"] = [0, 1, 2] if have_mpi else [1, 2]
    if have_mpi:
        try:
            import mpi4py

            info["mpi4py"] = mpi4py.__version__
        except ImportError:
            info["mpi4py"] = None
            info["notes"].append("mpi4py missing (an MPI build needs it)")

    # What DistStore(comm=None) and method=None resolve to (torch.py:_resolve_comm).
    comm = os.environ.get("DDSTORE_COMM", "").lower()
    if not comm:
        comm = "mpi" if have_mpi else ("torch" if torch_dist_init else "file")
    info["comm_none_resolves_to"] = comm
    info["default_method"] = int(
        os.environ.get("DDSTORE_METHOD", "0" if comm == "mpi" else "1")
    )
    info["handshake_dir"] = os.environ.get("DDSTORE_HANDSHAKE_DIR", "./ddstore_hs")
    if comm == "torch" and not torch_dist_init:
        info["notes"].append(
            "DDSTORE_COMM=torch but torch.distributed is not initialized here"
        )

    info["env_set_by_user"] = user_env
    info["env_after_import"] = {v: os.environ.get(v) for v in FABRIC_VARS}
    if (
        user_env["FI_MR_CACHE_MONITOR"] == "kdreg2"
        and user_env["DDSTORE_FORK_WORKERS"] != "1"
    ):
        info["notes"].append(
            "kdreg2 fails in forked DataLoader workers; set DDSTORE_FORK_WORKERS=1 to use them"
        )
    if os.environ.get("SLURM_JOB_ID") and not user_env["DDSTORE_NIC_MAP"]:
        info["notes"].append(
            "DDSTORE_NIC_MAP unset: export it once in the job script (hpc.md)"
        )

    info["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    info["system"] = os.environ.get("NERSC_HOST") or os.environ.get("LMOD_SYSTEM_NAME")
    cxi = "/sys/class/cxi"
    info["cxi_devices"] = sorted(os.listdir(cxi)) if os.path.isdir(cxi) else []
    info["srun"] = shutil.which("srun")
    info["notes"].append(
        "methods_in_build is what was compiled; DDStore's test/test_single.py checks they work"
    )
    return info


def main():
    info = collect()
    if "--json" in sys.argv[1:]:
        print(json.dumps(info, indent=1))
        return 0 if info.get("pyddstore") else 1
    notes = info.pop("notes")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k:28s} {v if v is not None else '(unset)'}")
        else:
            print(f"{key:30s} {value if value is not None else '-'}")
    for n in notes:
        print(f"note: {n}")
    return 0 if info.get("pyddstore") else 1


if __name__ == "__main__":
    sys.exit(main())
