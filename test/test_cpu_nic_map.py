"""cpu_nic_map topology query: fake sysfs trees modeled on several systems,
plus a live comparison with hwloc-calc where both are available.

Run: python -m pytest -q test/test_cpu_nic_map.py   (no MPI needed)
"""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cpu_nic_map  # noqa: E402


def cpus(*ranges):
    out = set()
    for lo, hi in ranges:
        out.update(range(lo, hi + 1))
    return out


def fmt(pus):
    return cpu_nic_map.compress_ranges(pus)


def make_sysfs(root, online, nodes, nics):
    """online: set of PUs; nodes: {numa: PUs}; nics: {name: (numa, PUs)}."""
    os.makedirs(root / "devices/system/cpu")
    (root / "devices/system/cpu/online").write_text(fmt(online) + "\n")
    for n, pus in nodes.items():
        d = root / f"devices/system/node/node{n}"
        os.makedirs(d)
        d.joinpath("cpulist").write_text(fmt(pus) + "\n")
    os.makedirs(root / "class/net/lo")  # no device/: skipped
    for name, (numa, pus) in nics.items():
        d = root / f"class/net/{name}/device"
        os.makedirs(d)
        d.joinpath("numa_node").write_text(f"{numa}\n")
        d.joinpath("local_cpulist").write_text(fmt(pus) + "\n")


def env_map(monkeypatch, root, provider="hsn"):
    monkeypatch.setattr(cpu_nic_map, "SYSFS", str(root))
    monkeypatch.setattr(
        cpu_nic_map, "hcalc", lambda *a: pytest.fail("hwloc-calc called")
    )
    return cpu_nic_map.parse_env(cpu_nic_map.serialize_env(provider=provider))


def test_frontier(tmp_path, monkeypatch):
    # 64 cores x 2 threads, 4 NUMA nodes, one NIC per NUMA node.
    nodes = {
        n: cpus((16 * n, 16 * n + 15), (64 + 16 * n, 79 + 16 * n)) for n in range(4)
    }
    nics = {
        "hsn2": (0, nodes[0]),
        "hsn1": (1, nodes[1]),
        "hsn3": (2, nodes[2]),
        "hsn0": (3, nodes[3]),
    }
    make_sysfs(tmp_path, cpus((0, 127)), nodes, nics)
    m = env_map(monkeypatch, tmp_path, provider="cxi")
    assert cpu_nic_map.serialize_env(provider="cxi") == (
        "cxi0=48-63,112-127;cxi1=16-31,80-95;cxi2=0-15,64-79;cxi3=32-47,96-111"
    )
    assert m == cpu_nic_map.parse_env(cpu_nic_map.serialize_env(provider="cxi"))


def test_perlmutter_gpu(tmp_path, monkeypatch):
    # 64 cores x 2 threads (NPS4: 4 NUMA nodes), 4 NICs, one per NUMA node.
    nodes = {
        n: cpus((16 * n, 16 * n + 15), (64 + 16 * n, 79 + 16 * n)) for n in range(4)
    }
    nics = {f"hsn{n}": (3 - n, nodes[3 - n]) for n in range(4)}
    make_sysfs(tmp_path, cpus((0, 127)), nodes, nics)
    m = env_map(monkeypatch, tmp_path)
    assert set(m) == cpus((0, 127))
    for n in range(4):
        assert {c for c, nic in m.items() if nic == f"hsn{n}"} == nodes[3 - n]


def test_perlmutter_cpu(tmp_path, monkeypatch):
    # 2 x 64 cores x 2 threads, 8 NUMA nodes, a single NIC on NUMA node 1:
    # every CPU maps to it, including those on other NUMA nodes.
    nodes = {
        n: cpus((16 * n, 16 * n + 15), (128 + 16 * n, 143 + 16 * n)) for n in range(8)
    }
    make_sysfs(tmp_path, cpus((0, 255)), nodes, {"hsn0": (1, nodes[1])})
    m = env_map(monkeypatch, tmp_path)
    assert m == {c: "hsn0" for c in range(256)}


def test_aurora(tmp_path, monkeypatch):
    # 2 sockets x 52 cores x 2 threads, 8 NICs, 4 per socket, all four
    # reporting the socket's CPUs: each NIC gets a distinct share of cores
    # and of their hyperthreads. NUMA nodes 2/3 (HBM) have no CPUs or NICs.
    s0, s1 = cpus((0, 51), (104, 155)), cpus((52, 103), (156, 207))
    nodes = {0: s0, 1: s1, 2: set(), 3: set()}
    nics = {f"hsn{i}": (0, s0) for i in range(4)}
    nics.update({f"hsn{i}": (1, s1) for i in range(4, 8)})
    make_sysfs(tmp_path, cpus((0, 207)), nodes, nics)
    m = env_map(monkeypatch, tmp_path)
    assert set(m) == cpus((0, 207))
    for i in range(8):
        mine = {c for c, nic in m.items() if nic == f"hsn{i}"}
        assert mine <= (s0 if i < 4 else s1)
        assert len(mine & cpus((0, 103))) == 13  # cores
        assert len(mine & cpus((104, 207))) == 13  # hyperthreads


def test_no_numa_info(tmp_path, monkeypatch):
    # VM-like: numa_node -1, no node directories; 2 NICs with disjoint
    # local CPUs, and CPUs outside both go to the first NIC.
    nics = {"hsn0": (-1, cpus((0, 3))), "hsn1": (-1, cpus((4, 7)))}
    make_sysfs(tmp_path, cpus((0, 9)), {}, nics)
    m = env_map(monkeypatch, tmp_path)
    assert m == {
        **{c: "hsn0" for c in range(4)},
        **{c: "hsn1" for c in range(4, 8)},
        8: "hsn0",
        9: "hsn0",
    }


def test_hwloc_fallback(tmp_path, monkeypatch):
    # No NUMA cpulist in sysfs: falls back to hwloc-calc.
    nodes = {0: cpus((0, 3)), 1: cpus((4, 7))}
    make_sysfs(
        tmp_path, cpus((0, 7)), {}, {"hsn0": (0, nodes[0]), "hsn1": (1, nodes[1])}
    )
    monkeypatch.setattr(cpu_nic_map, "SYSFS", str(tmp_path))
    calls = []

    def hcalc(loc, itype):
        calls.append(loc)
        return sorted(cpus((0, 7)) if loc == "all" else nodes[int(loc.split(":")[1])])

    monkeypatch.setattr(cpu_nic_map, "hcalc", hcalc)
    assert cpu_nic_map.serialize_env() == "hsn0=0-3;hsn1=4-7"
    assert calls == ["all", "NUMA:0", "NUMA:1"]


def test_matches_hwloc():
    """On a real node with hsn NICs and hwloc-calc: sysfs and hwloc-calc
    give the same map."""
    import glob

    if not glob.glob("/sys/class/net/hsn*/device") or not shutil.which("hwloc-calc"):
        pytest.skip("needs hsn* NICs and hwloc-calc")
    sysfs = cpu_nic_map.serialize_env()
    orig = cpu_nic_map._read_cpulist
    try:
        cpu_nic_map._read_cpulist = lambda p: (
            None if "/devices/system/" in p else orig(p)
        )
        hwloc = cpu_nic_map.serialize_env()
    finally:
        cpu_nic_map._read_cpulist = orig
    assert sysfs == hwloc
