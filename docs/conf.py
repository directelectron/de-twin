# Sphinx configuration for de-twin.

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# -- Project information -----------------------------------------------------
project = "de-twin"
copyright = "2026, Carter Francis"
author = "Carter Francis"
# The single source of the version is pyproject.toml (bumped by Prepare Release).
release = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
version = release

# In CI the Docs workflow sets DOCS_VERSION to the tag ("v0.1.0") or "dev".
_docs_version = os.environ.get("DOCS_VERSION", "dev")
_base = "https://directelectron.github.io/de-twin/"
html_baseurl = f"{_base}{_docs_version}/"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Docstrings reference many short member names across classes; keep genuinely missing
# references failing (nitpicky stays off) but quiet the ambiguous-target warnings.
suppress_warnings = ["ref.python", "myst.header"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "orix": ("https://orix.readthedocs.io/en/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_root",
    "Thumbs.db",
    ".DS_Store",
    "auto_examples/*.ipynb",
    "auto_examples/*.py",
    "auto_examples/*.zip",
    "auto_examples/*.json",
    "auto_examples/*.md5",
    "sg_execution_times.rst",
]

# -- Sphinx Gallery ----------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"[\\/]plot_",
    "plot_gallery": True,
    "download_all_examples": False,
    "remove_config_comments": True,
    "image_scrapers": ("matplotlib",),
    "abort_on_example_error": True,
    "first_notebook_cell": None,
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "de-twin"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/directelectron/de-twin",
    "logo": {"text": "de-twin"},
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": f"{_base}switcher.json",
        "version_match": _docs_version,
    },
    "check_switcher": False,
    "show_toc_level": 2,
}
