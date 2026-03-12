# Installation Guide
This section provides a step-by-step guide for installing Acherus package, aimed at both users and developers.

The installation guide is devoted to UNIX-based operating systems Linux and macOS---since they are dominant in scientific computing---from personal basic applications (Linux/macOS-based) to supercomputers, clusters, and high performance computing (HPC) facilities (entirely Linux-based). However, Acherus ecosystem is entirely dependent on Python libraries and modules, hence it can be used in any Linux, macOS, or Windows operating system. In the latter case, there are two approaches. The first one is to give up on Windows and use a UNIX machine instead. The second one is to install a Linux distribution on a Windows machine, which is easy to do following the [WSL Guide](https://learn.microsoft.com/en-us/windows/wsl/install).

```{contents}
:depth: 3
```

## Acherus installation

### For users
Acherus is now part of the [PyPI](https://pypi.org/project/acherus) software repository and can be installed as a package using `pip`:
```{code-block} console
pip install acherus
```

Once installed, to upgrade to the latest version simply do
```bash
pip install --upgrade acherus
```

### For developers
The development and management of Acherus is carried through [GitHub](https://github.com/ismatorresgarcia/acherus). Some developers might prefer to use the [GitHub CLI](https://cli.github.com) to contribute straight from their terminal.

To start using Acherus and access the various ecosystem files, just clone the repository by choosing one of the two options below:
```bash
# SSH (password-protected key):
git clone git@github.com:ismatorresgarcia/acherus.git

# HTTPS (web URL):
git clone https://github.com/ismatorresgarcia/acherus.git
```

Next, on a previously created Python virtual environment, proceed to install the dependencies. Move to `acherus/` project directory and install them by doing:
```bash
cd acherus
pip install -r requirements.txt
```
For testing and experimenting with upcoming features, installing the cloned package in editable mode allows developers to see their new changes without reinstalling the package. It also installs automatically all the dependencies through the `pyproject.toml` configuration file. Move to `acherus/` project directory and install it in editable mode by doing:
```bash
cd acherus
pip install -e .
```
If the developer has any ideas, suggestions, or improvements, and wants to push them to Acherus, the recommended route is to create a [fork](https://github.com/ismatorresgarcia/acherus/fork) from Acherus `main` [branch](https://github.com/ismatorresgarcia/acherus) in your own GitHub repository. Then, create a new branch, and submit a pull request (PR). Below are listed the five steps to follow:

1. Fork the repository: <https://github.com/ismatorresgarcia/acherus/fork>
2. Create a new branch:
```bash
git checkout -b <branch-name>
```
3. Make your changes and commit them:
```bash
git add <file-name>.py
git commit -m '<changes-description>'
```
4. Push to the new branch:
```bash
git push origin <branch-name>
```
5. Submit a pull request: <https://github.com/ismatorresgarcia/acherus/pulls>

## Dependencies
As mentioned at the beginning, Acherus is an entirely Python-written code which depends on a few well-established third-party Python libraries:
* [NumPy](https://numpy.org): Fundamental package for scientific computing in Python. Specially important for fast operations on arrays of multidimensional shapes.
* [SciPy](https://scipy.org): Mathematical algorithms and high-level functions built on NumPy. Specially important for accessing various numerical linear algebra tools.
* [Matplotlib](https://matplotlib.org): Comprehensive library for creating static, animated, and interactive figures in Python. Specially useful for plotting and animating 1D and 2D plots.
* [H5Py](https://www.h5py.org): Python interface to the HDF5 binary data format. Specially important for storing laser field intensity and electron density 3D arrays at different propagation steps.
* [Numba](https://numba.pydata.org): Performance library that uses JIT compilation to translate on runtime Python and NumPy pieces of code into machine code. Specially important for enhancing the speed of sequential algorithms from explicit numerical schemes.
* [pyFFTW](https://numba.pydata.org): Python wrapper around the well-known C subroutine library [FFTW](https://www.fftw.org) for computing the _Fastest Fourier Transform in the West_. Specially important for implementing the high-speed FFT algorithms and low-level controls in pseudospectral schemes.

These are the dependencies required by Acherus. When installing Acherus from Pip, the dependencies download and installation is done together with the package. Either way, the list of dependencies can be explicitly installed in a chosen Python virtual environment moving to `acherus/` project directory and typing in the terminal:
```
pip install -r requirements.txt
```

## Python installation
The first step is to know if the machine does not have a Python installation setup. Not having a system-wide Python installed is incredibly rare, since many low-level tools depend on Python for running scripts. However, some minimal systems might not have Python installed at all. To see if your system has some internal Python version installed, type in your terminal:
```bash
# check for modern setups
python3 --version
# check for older setups
python3 --version
```
If a version number such as
```console
Python X.YY.Z
```
is shown in the terminal, do **not upgrade or replace it manually** and move to the [version installation](version_installation) section. Otherwise, follow these brief guideline for installing Python 3 in [Linux](linux_installation), [macOS](macos_installation), and [Windows](windows_installation) operating systems:
```{note}
The system-wide version in the machine is not really meant to be used for anything besides basic operating system infrastructure and maybe learning purposes. Furthermore, the version shipped might be deprecated or out of date from the latest official [Python](https://www.python.org/downloads/) release, which is considered the production version for any active user.
```

(linux_installation)=
### Linux
Most popular Linux distributions already include Python because the package manager needs it to execute some scripts. If not installed, read the distribution documentation for the basic package manager instructions, commands, and available distribution repositories. Once this is done, simply search in the repository for the latest stable name for Python and pass it as an argument to the package manager system for installation. Here are some examples:

1. Debian-based ([Ubuntu](https://ubuntu.com/server/docs/how-to/software/package-management/), [Debian](https://www.debian.org/doc/manuals/debian-reference/ch02.en.html), [Linux Mint](https://linuxmint.com/documentation.php))
```bash
sudo apt-get install <python-version-name>
```
2. Red Hat-based ([RHEL](https://www.redhat.com/en/blog/install-software-packages-rhel), [CentOS](https://www.centos.org), [Fedora](https://docs.fedoraproject.org/en-US/quick-docs/dnf/))
```bash
sudo dnf install <python-version-name>
```
1. [Arch Linux](https://wiki.archlinux.org/title/Main_page) & [Manjaro](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://wiki.manjaro.org/index.php/Pacman_Overview&ved=2ahUKEwiC3s6n7JWTAxXWU6QEHTHfF4QQmuEJegQIEBAB&usg=AOvVaw2nTLel5yOCIfTIbB2xTVof)
```bash
sudo pacman -S <python-version-name>
```
4. [OpenSUSE](https://doc.opensuse.org/documentation/leap/reference/html/book-reference/cha-sw-cl.html)
```bash
sudo zypper install <python-version-name>
```

Then, verify the installation in the terminal:
```bash
python3 --version
pip3 --version
```

(macos_installation)=
### macOS
Newer versions of macOS may not have Python installed for users (older versions used to have Python 2.X by default). In this case, the simplest approach is to download a recent and stable release from the [Python](https://www.python.org/downloads/macos/) website and follow the installer instructions. In the process, Python should be added automatically to the `PATH` during the configuration and bundled together with the `pip` installation.

Then, check both:
```bash
python3 --version
pip3 --version
```

(windows_installation)=
### Windows
If Python is not installed on a Windows machine (and assuming the user wants to stick with Windows, instead of installing the [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)), install the latest version of Python from the [Python](https://www.python.org/downloads/windows/) website and follow the installer instructions. Remember during installation to add Python to `PATH` and allow the installer to include `pip` by checking the corresponding options.

Then, verify the installation:
```bash
python3 --version
pip3 --version
```

(version_installation)=
## Version installation
Once verified the machine has installed a system-wide Python version, the next step is to install [Pyenv](https://github.com/pyenv/pyenv) and follow the configuration instructions for your operating system. Pyenv is a simple, lightweight, and easy to use tool that simplifies the **management** of multiple Python versions on a system. You can install different Python versions and switch between them as needed. Before installing Pyenv, make sure the necessary dependencies, such as [Git](https://git-scm.com/install/), are installed.

To show all Python versions within Pyenv, do:
```bash
pyenv install --list
```
To install any of the Python versions available, do:
```bash
pyenv install <X.YY.Z>
```

One of the main advantages of Pyenv is it downloads, builds, installs, and sets multiple Python versions **completely separate** from the system Python. This avoid any conflict between previously existing Python versions and the possibility of interfering with the system installation, which would become a nightmare and potentially break the system. For example, let's say Python 3.12.3 was installed. To set it as the default (global) version of the system, just do:
```bash
pyenv global 3.12.3
```
Another interesting thing, quite useful for developers, is the ability to set one version per project. For example, to set a project-specific (local) Python 3.12.3, do inside the `acherus/` project folder:
```
pyenv local 3.12.3
```
This simply creates a `.python-version` file inside the project folder, meaning everything contained in it (without affecting anything outside) use that Python version automatically. It guarantees the Python version being used when running any simulation is restricted to the `acherus/` project folder, as well as shifting easily between other versions. Also, Pyenv shims, routes, and unifies every active Python version under the commands `python` and `pip`, so there is no need to type `python3` and `pip3` anymore.

(virtual_environment)=
## Environment isolation
Once installed and selected a Python version using Pyenv, the final strongly recommended step is to create a **virtual environment** for the project. The reason why virtual environments are crucial is to **isolate** the installation of Python packages and third-party libraries without interfering with other projects or the system Python. Just as Pyenv avoids conflicts between different Python versions, virtual environments prevent conflicts between different packages and/or their versions. For example, different projects may require different versions of the same package, which is one of the reasons why developers use them on a daily basis.

To create and activate a virtual environment inside your project folder, do (Windows requires giving execution permissions first):
::::{tab-set}

:::{tab-item} Linux & macOS
```bash
python -m venv .<venv-name>
source .<venv-name>/bin/activate
```
:::

:::{tab-item} Windows (CMD)
```bash
python -m venv .<venv-name>
.<venv-name>\Scripts\activate.bat
```
:::

:::{tab-item} Windows (PowerShell)
```bash
python -m venv .<venv-name>
.<venv-name>\Scripts\Activate.ps1
```

:::

::::

Instead of creating a `.<venv-name>` inside every project, the user might choose to store virtual environments in some centralized location. This can make management easier if you work with many projects. For example, a global folder named `virtualenvs/` in your home directory could host a new environment:
```bash
python -m venv ~/virtualenvs/<venv-name>
```

Once the environment is activated, any `pip` command installs **exclusively inside the virtual environment**, separated from the system Python or other project folders:
```bash
pip install <package-name>
```
For example, to recreate the exact environment used in the `acherus/` project:
```bash
pip install -r requirements.txt
```
When finished, to return the shell to the previous state, type:
```bash
deactivate
```
In summary, we deeply encourage users and developers to set a Python 3 version (between 3.10 and 3.13) with Pyenv and create a dedicated virtual environment:
```bash
# 1. Set one python version with pyenv
pyenv local <3.YY.Z> # Option 1: project folder
pyenv global <3.YY.Z> # Option 2: entire system

# 2. Set one python virtual environment (Linux & macOS)
source .<venv-name>/bin/activate # Option 1: project folder
source ~/virtualenvs/<venv-name>/bin/activate # Option 2: shared folder

# 3. Set acherus installation with pip
pip install acherus
```

