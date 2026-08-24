#!/usr/bin/env python3
"""CPU <-> nearest HSN (Slingshot) NIC topology, and FABRIC_IFACE auto-selection.

Requires: module load hwloc  (or lstopo/hwloc-calc on PATH)

Three layers, one file:
  - build_map()/serialize_env()/parse_env(): pure hwloc-calc topology query,
    independent of any particular process's CPU affinity.
  - allocated_nics(): affinity-aware wrapper — which NIC(s) is *this
    process*, given its actual pinning (os.sched_getaffinity), closest to.
  - select_fabric_iface(): called automatically by PyDDStore.__cinit__
    (src/pyddstore.pyx) for method=1/2 to set FABRIC_IFACE if not already set.

CLI:
  cpu_nic_map.py                print the full CPU -> nearest HSN NIC table
  cpu_nic_map.py 42              print only the nearest HSN NIC for cpu 42
  cpu_nic_map.py -p 'ens*' 0     match a different NIC name pattern
  cpu_nic_map.py --env           print the compact DDSTORE_NIC_MAP env-var value
  cpu_nic_map.py --allocated     print this process's allocated CPUs and nearest NIC(s)

  export DDSTORE_NIC_MAP=$(python3 cpu_nic_map.py --env)
  srun --threads-per-core=2 -n8 -c14 python cpu_nic_map.py --allocated
"""
import argparse
import fnmatch
import glob
import os
import subprocess
import sys


def hcalc(loc, itype):
    # --disallowed (must come first): include cores excluded by SLURM core
    # specialization (e.g. cpu 0, 8, 16, ...) so the map covers all 128
    # CPUs, not just the ~112 currently allocatable to jobs -- a rank's
    # affinity mask isn't guaranteed to avoid them (e.g. -S0 jobs).
    # -p: report physical (OS/kernel) indices, matching os.sched_getaffinity()
    # ids. Without it, hwloc-calc reports logical indices, which silently
    # diverge from real CPU ids whenever core specialization or other
    # exclusions shift the logical numbering (see lstopo's "P#" vs "L#").
    out = subprocess.run(
        ["hwloc-calc", "--disallowed", "-p", "-I", itype, loc],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.decode().strip()
    return [int(x) for x in out.split(",")] if out else []


def build_map(pattern):
    nics = sorted(os.path.basename(p) for p in glob.glob("/sys/class/net/*")
                  if os.path.exists(os.path.join(p, "device"))
                  and fnmatch.fnmatch(os.path.basename(p), pattern))
    if not nics:
        sys.exit(f"no NICs matching '{pattern}' found under /sys/class/net")

    nic_closest = {n: set(hcalc(f"os={n}", "PU")) for n in nics}
    nic_numa = {n: (hcalc(f"os={n}", "NUMA") or [None])[0] for n in nics}
    if not any(nic_closest.values()):
        sys.exit(
            "hwloc-calc found no PUs local to any NIC (os=<name> lookups came back "
            "empty) -- this process's hwloc topology view has no visibility into the "
            "NICs, so any nearest-NIC answer would be a silent guess, not real data. "
            "This has been observed when running directly inside a plain `srun` task; "
            "try running from the sbatch batch step's own shell instead."
        )

    # multiple NICs can share a NUMA node; pick the numerically/PCI-closest
    # NIC as the "same-NUMA fallback owner" for cores not exactly local to any NIC
    numa_to_nics = {}
    for n, numa in nic_numa.items():
        numa_to_nics.setdefault(numa, []).append(n)

    all_pus = hcalc("all", "PU")
    pu_numa = {}
    for numa in numa_to_nics:
        for pu in hcalc(f"NUMA:{numa}", "PU"):
            pu_numa[pu] = numa

    def nearest(pu):
        numa = pu_numa.get(pu)
        exact_owner = next((n for n in nics if pu in nic_closest[n]), None)
        if exact_owner:
            return exact_owner, "yes", numa
        candidates = numa_to_nics.get(numa, nics)
        return candidates[0], "same-NUMA", numa

    return all_pus, nearest


def compress_ranges(values):
    values = sorted(values)
    out = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        out.append(str(values[i]) if i == j else f"{values[i]}-{values[j]}")
        i = j + 1
    return ",".join(out)


def serialize_env(pattern="hsn*"):
    all_pus, nearest = build_map(pattern)
    by_nic = {}
    for pu in all_pus:
        by_nic.setdefault(nearest(pu)[0], []).append(pu)
    return ";".join(f"{nic}={compress_ranges(pus)}" for nic, pus in sorted(by_nic.items()))


def parse_env(s):
    cpu_to_nic = {}
    for segment in s.split(";"):
        nic, ranges = segment.split("=", 1)
        for r in ranges.split(","):
            if "-" in r:
                lo, hi = r.split("-")
                cpu_to_nic.update((cpu, nic) for cpu in range(int(lo), int(hi) + 1))
            else:
                cpu_to_nic[int(r)] = nic
    return cpu_to_nic


def allocated_nics(pattern="hsn*", nic_map=None):
    """Which NIC(s) this process's actual CPU affinity is nearest to.

    Uses os.sched_getaffinity(0) to get the real CPU set the process is
    bound to (respects SLURM --cpus-per-task/--cpu-bind and cgroups).

    nic_map: an explicit precomputed map string (see serialize_env()/--env
    format), taking priority over the DDSTORE_NIC_MAP env var. Pass this
    when the caller already has the map from somewhere other than the
    process environment. Falls back to a live hwloc-calc query when neither
    is available.

    Prefer a precomputed map: hwloc-calc's NIC/PCI visibility has been
    observed to fail silently inside some srun tasks, so computing the map
    once (e.g. in the sbatch batch step's own shell, where NIC visibility is
    reliable) and sharing it -- via DDSTORE_NIC_MAP or the nic_map argument
    -- is more robust than querying hwloc fresh in every rank.
    """
    allocated = sorted(os.sched_getaffinity(0))
    map_str = nic_map if nic_map is not None else os.environ.get("DDSTORE_NIC_MAP")
    if map_str:
        cpu_to_nic = parse_env(map_str)
        nics = {cpu_to_nic[c] for c in allocated if c in cpu_to_nic}
    else:
        all_pus, nearest = build_map(pattern)
        nics = {nearest(c)[0] for c in allocated if c in all_pus}
    return allocated, nics


def select_fabric_iface(nic_map=None):
    """Pick the libfabric NIC (FABRIC_IFACE) for this process, if not
    already set. Defers to allocated_nics(), which uses nic_map when given,
    else DDSTORE_NIC_MAP when set, or a live hwloc-calc/lstopo query
    (build_map) against this process's real CPU affinity when neither is.

    Called automatically by PyDDStore.__cinit__ (src/pyddstore.pyx) for
    method=1/2.

    nic_map: an explicit precomputed map string (serialize_env()/--env
    format), for callers that already have the map from somewhere other
    than the process environment.
    """
    if "FABRIC_IFACE" in os.environ:
        return os.environ["FABRIC_IFACE"]

    allocated, nics = allocated_nics(nic_map=nic_map)
    if not nics:
        raise RuntimeError(
            f"could not determine a nearest HSN NIC for this rank's CPU "
            f"affinity {allocated}; set FABRIC_IFACE explicitly to work "
            f"around this"
        )
    iface = sorted(nics)[0]
    if len(nics) > 1:
        print(f"FABRIC_IFACE: affinity spans {sorted(nics)}, picking {iface}")
    os.environ["FABRIC_IFACE"] = iface
    return iface


def main():
    parser = argparse.ArgumentParser(
        description="Find the nearest HSN (Slingshot) NIC for a given CPU (PU) id, "
                     "based on hwloc PCI/NUMA locality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  cpu_nic_map.py           print the full CPU -> nearest HSN NIC table\n"
            "  cpu_nic_map.py 42        print only the nearest HSN NIC for cpu 42\n"
            "  cpu_nic_map.py -p 'ens*' 0   match a different NIC name pattern\n"
            "  export DDSTORE_NIC_MAP=$(cpu_nic_map.py --env)   compute once, share via env\n"
            "  srun ... python cpu_nic_map.py --allocated   show this task's allocated CPUs + nearest NIC(s)\n"
        ),
    )
    parser.add_argument("cpu", nargs="?", type=int,
                         help="CPU (PU) id to look up; omit to print the full table")
    parser.add_argument("-p", "--pattern", default="hsn*",
                         help="glob pattern for NIC names to consider (default: hsn*)")
    parser.add_argument("--env", action="store_true",
                         help="print only the compact DDSTORE_NIC_MAP env-var value")
    parser.add_argument("--allocated", action="store_true",
                         help="print this process's allocated CPUs (os.sched_getaffinity) "
                              "and their nearest NIC(s), instead of the full table")
    args = parser.parse_args()

    if args.env:
        print(serialize_env(args.pattern))
        return

    if args.allocated:
        allocated, nics = allocated_nics(args.pattern)
        print(f"allocated CPUs: {allocated}")
        print(f"nearest NIC(s): {sorted(nics)}")
        return

    all_pus, nearest = build_map(args.pattern)

    if args.cpu is not None:
        if args.cpu not in all_pus:
            sys.exit(f"cpu id {args.cpu} not found (valid range: {min(all_pus)}-{max(all_pus)})")
        owner, exact, numa = nearest(args.cpu)
        print(owner)
        return

    print(f"{'CPU':>4}  {'NUMA':>4}  {'nearest NIC':>11}  exact")
    for pu in all_pus:
        owner, exact, numa = nearest(pu)
        print(f"{pu:>4}  {numa!s:>4}  {owner:>11}  {exact}")


if __name__ == "__main__":
    main()
