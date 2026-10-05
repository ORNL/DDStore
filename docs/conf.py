"""Sphinx configuration for the DDStore documentation (MyST Markdown)."""

import os
import sys

# Document pyddstore from the source tree; its compiled core, MPI and torch
# are mocked, so the docs build needs neither a compiler, MPI nor libfabric.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "DDStore"
author = "Oak Ridge National Laboratory"
copyright = "UT-Battelle, LLC"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

# GitHub-style anchors on headings, so links like page.md#get_batchname-arr-indices work
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]

autodoc_mock_imports = ["mpi4py", "torch", "pyddstore._core"]
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "furo"
html_title = "DDStore"
html_logo = "../images/DDStore-logo.png"
html_theme_options = {
    "source_repository": "https://github.com/ORNL/DDStore/",
    "source_branch": "main",
    "source_directory": "docs/",
}
