"""Information file for the Acherus package."""

import re
from pathlib import Path

from setuptools import find_packages, setup

# Get package version
version_file = Path(__file__).parent / "acherus/_version.py"
# Open the version file
with open(version_file, "r", encoding="utf-8") as f:
    version_info = f.read()
# Search for the "__version__" pattern
version_find = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", version_info)
# Extract the version string
__version__ = version_find.group(1) if version_find else "0.0.0"

# Get full description from README.md
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements.txt for extras_require
with open("requirements.txt", encoding="utf-8") as f:
    plotting_required = f.read().splitlines()

setup(
    name="acherus",
    version=__version__,
    description="3D spatiotemporal solver, focused in atmospheric laser-plasma filaments",
    keywords=[
        "simulation",
        "laser",
        "filamentation",
        "atmospheric",
        "plasma",
        "oxygen",
        "nitrogen",
        "femtosecond",
        "picosecond",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://acherus.readthedocs.io/",
    author="Ismael Torres García et al.",
    author_email="i.torresg@upm.es",
    # download_url="https://pypi.python.org/pypi/acherus",
    project_urls={
        "Bug Tracker": "https://github.com/ismatorresgarcia/acherus/issues",
        # "Documents": "https://acherus.readthedocs.io/en/latest/index.html",
        "Source Code": "https://github.com/ismatorresgarcia/acherus",
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics :: Atmospheric Science",
    ],
    entry_points={
        "console_scripts": [
            "acherus = acherus.main:main",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "numpy<2.0",
        "scipy",
        "h5py",
        "numba",
    ],
    extras_require={
        "plotting": plotting_required,
    },
    # tests_require=["pytest"],
)
