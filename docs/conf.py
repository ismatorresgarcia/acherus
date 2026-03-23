"""Sphinx configuration file."""

import os


def read_svg(filename):
    path = os.path.join(os.path.dirname(__file__), "_static/images", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# -- Project information -----------------------------------------------------
project = "acherus"
project_copyright = "2026, IFN, Universidad Politécnica de Madrid"
author = "Ismael Torres García, Eduardo Oliva Gonzalo"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",  # for math
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    # External
    "myst_parser",  # for markdown
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
]

templates_path = ["_templates"]

# MyST Markdown extensions.
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"  # table of contents master document

# Pygments (Python-driven syntax highlighting) style.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "Acherus"

html_static_path = ["_static"]
html_logo = "_static/images/acherus-logo-g.png"

# -- Furo Theme options ---------------------------------------------
html_theme_options = {
    "sidebar_hide_name": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ismatorresgarcia/acherus",
            "html": read_svg("github-logo.svg"),
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/acherus",
            "html": read_svg("pypi-logo.svg"),
            "class": "",
        },
    ],
}
