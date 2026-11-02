"""skills/ddstore matches the code: front matter, links, the checker, and
the names and signatures api.md documents. One process, no MPI needed."""

import inspect
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(ROOT, "skills", "ddstore")
DOCS = [
    os.path.join(dirpath, f)
    for dirpath, _, files in os.walk(os.path.join(ROOT, "skills"))
    for f in files
    if f.endswith(".md")
]


def read(path):
    with open(path) as fh:
        return fh.read()


def test_front_matter():
    m = re.match(r"---\n(.*?)\n---\n", read(os.path.join(SKILL, "SKILL.md")), re.S)
    assert m, "SKILL.md must start with YAML front matter"
    fields = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    assert fields["name"].strip() == "ddstore"
    assert 50 < len(fields["description"].strip()) <= 1024


def _anchors(text):
    # GitHub-style heading anchors
    out = set()
    for h in re.findall(r"^#+ (.+)$", text, re.M):
        a = re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")
        out.add(a)
    return out


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: os.path.relpath(p, ROOT))
def test_links(doc):
    for target in re.findall(r"\]\(([^)\s]+)\)", read(doc)):
        if re.match(r"[a-z]+://", target):
            continue
        path, _, anchor = target.partition("#")
        full = (
            os.path.normpath(os.path.join(os.path.dirname(doc), path)) if path else doc
        )
        assert os.path.exists(full), f"{doc}: broken link {target}"
        if anchor and full.endswith(".md"):
            assert anchor in _anchors(read(full)), f"{doc}: no anchor {target}"


def test_check_env():
    script = os.path.join(SKILL, "scripts", "check_env.py")
    out = subprocess.run(
        [sys.executable, script, "--json"], capture_output=True, text=True, cwd="/"
    )
    assert out.returncode == 0, out.stderr
    info = json.loads(out.stdout)
    for key in (
        "have_mpi",
        "methods_in_build",
        "comm_none_resolves_to",
        "default_method",
    ):
        assert key in info


def _documented_calls(text):
    """name -> parameter names, from the code blocks of api.md."""
    calls = {}
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        for name, params in re.findall(r"^(\w+)\(([^)]*)\)", block, re.M | re.S):
            names = [p.split("=")[0].strip() for p in params.split(",") if p.strip()]
            calls[name] = [n for n in names if not n.startswith("**")]
    return calls


def test_api_signatures():
    pt = pytest.importorskip("pyddstore.torch")
    calls = _documented_calls(read(os.path.join(SKILL, "api.md")))
    checked = 0
    for name, params in calls.items():
        obj = getattr(pt, name, None)
        if obj is None:
            continue  # PyDDStore methods etc. are not pyddstore.torch callables
        real = list(inspect.signature(obj).parameters)
        real = [p for p in real if p not in ("self", "kwargs")]
        assert (
            real[: len(params)] == params
        ), f"{name}: documented {params}, real {real}"
        checked += 1
    assert checked >= 6


def test_api_attributes():
    pt = pytest.importorskip("pyddstore.torch")
    text = read(os.path.join(SKILL, "api.md")) + read(
        os.path.join(SKILL, "patterns.md")
    )
    for attr in set(re.findall(r"`ds\.(\w+)", text)):
        assert hasattr(pt.DistDataset, attr) or attr in (
            "store",
            "ddstore",
        ), f"ds.{attr}"
    for attr in set(re.findall(r"`store\.(\w+)", text)):
        assert hasattr(pt.DistStore, attr) or attr in (
            "names",
            "method",
        ), f"store.{attr}"
