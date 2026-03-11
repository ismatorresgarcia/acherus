"""Sphinx configuration file."""

# -- Project information -----------------------------------------------------
project = "Acherus"
project_copyright = "2026, IFN, Universidad Politecnica de Madrid"
author = "Ismael Torres García, Eduardo Oliva Gonzalo"

# Short project version
version = "0.8"
# Full project version
release = "0.8.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",  # for eqs
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "myst_parser",  # for markdown
    "sphinx_design",
    "sphinx_last_updated_by_git",
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
pygments_dark_style = "monokai"

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- PyData Sphinx Theme options ---------------------------------------------
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "show_prev_next": True,
    "github_url": "https://github.com/ismatorresgarcia/acherus",
    "logo": {
        "image_light": "_static/images/acherus-logo-r.png",
        "image_dark": "_static/images/acherus-logo-g.png",
    },
}

html_context = {
    "github_user": "ismatorresgarcia",
    "github_repo": "acherus",
    "github_version": "main",
    "doc_path": "",
}

# Enable "Edit on GitHub" buttons
use_page_edit_button = True
