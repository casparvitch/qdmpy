Quantum Diamond MicroscoPy
==========================

<!-- MarkdownTOC -->

- [Installation](#installation)
    - [jupyterlab installation](#jupyterlab-installation)
    - [Saving widget state](#saving-widget-state)
    - [Exporting notebook to pdf](#exporting-notebook-to-pdf)
    - [Version Control](#version-control)
        - [Installation](#installation-1)
    - [Environment Management \(conda/pipenv\)](#environment-management-condapipenv)
    - [Gpufit](#gpufit)
    - [Linux](#linux)
- [Github/Gitlab details for GPUFit](#githubgitlab-details-for-gpufit)
- [Gpufit install](#gpufit-install)
    - [Methodology:](#methodology)
    - [Where to find these details:](#where-to-find-these-details)
    - [Installation Procedure](#installation-procedure)
    - [Building from source](#building-from-source)
- [Documentation](#documentation)

<!-- /MarkdownTOC -->


# Installation

### Jupyterlab installation

https://nodejs.org/en/
https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension

see: https://stackoverflow.com/questions/49542417/how-to-get-ipywidgets-working-in-jupyter-lab

### Saving widget state

In jupyterlab > settings > advanced settings editor > jupyter widgets > saveState: true

### Exporting notebook to pdf

- install [pandoc](https://pandoc.org/installing.html)
- need to change to %matplotlib inline?
	- check Github issues (e.g. [#16](https://github.com/matplotlib/ipympl/issues/16), [#150](https://github.com/matplotlib/ipympl/issues/150) and [#176](https://github.com/matplotlib/ipympl/pull/176))

### Version Control

The project is housed on [Gitlab](https://gitlab.unimelb.edu.au/sscholten/QDMPy), you will need to be given access by an owner and sign in with uni credentials. To communicate with the Gitlab server you will need to setup an ssh key (on each device connected, there will need to be one on each lab computer as well). My installation instructions below are taken from the Gitlab Docs [here](https://docs.gitlab.com/ee/ssh/).

You can also use Gitlab in-browser, i.e. not using the git commands at all. This is not recommended but can be great in a pinch.

Tip: It's a lot easier to read diffs/merges using the side-by-side visual. Somewhere in your diff tool/online it will have this options, give it a shot.


#### Installation


If you're on windows you will need to download a Git/OpenSSH client (Unix systems have it pre-installed). The simplest way to do this for integration with PyCharm is just to use [Git for Windows](https://gitforwindows.org/), even if you have WSL/Cygwin. Just do it, it isn't that big a package.

Open the Git Bash terminal (or Bash on Unix). Generate a new ED25519 SSH key pair:

```Bash
ssh-keygen -t ed25519 -C "<YOUR UNI EMAIL HERE>"
```

You will be prompted to input a file path to save it to. Just click Enter to put it in the default \~/.ssh/config file
Once that's decided you will be prompted to input a password. To skip press Enter twice (make sure you don't do this for a shared PC such as the Lab computers).

**Now add to your Gitlab account** (under settings in your browser)
To clip the public key to your clipboard for pasting, in Git Bash:

```Bash
cat \~/.ssh/id_ed25519.pub | clip
```

macOS:

```Bash
pbcopy < \~/.ssh/id_ed25519.pub
```

WSL/GNU/Linux (you may need to install xclip but it will give you instructions):

```Bash
xclip -sel clip < \~/.ssh/id_ed25519.pub
```

Now we can check whether that worked correctly:

```Bash
ssh -T git@git.unimelb.edu.au
```

If it's the first time you connect to Gitlab via SSH, you will be asked to verify it's authenticity (respond yes). You may also need to provide your password. If it responds with *Welcome to Gitlab, @username!* then you're all setup.


### Environment Management (conda/pipenv)


TODO


### Gpufit

Can install a binary from [here](https://github.com/gpufit/Gpufit/releases).

However this does not include the model functions we require (e.g. lorentzians, exponential decays) so you will need to write these into the source and rebuild it (each time you want a new model function...). Details below.

- [ ] Goal: keep our own branch with our model functions, rebase on master from time to time.
	- This way we can keep a group-wide collection of usable model functions
	- store in gitlab?
	- [multiple remotes, github](https://forum.sublimetext.com/t/working-with-multiple-remotes/53489/3)
- [ ] Keep in separate repo to QDMPy


- see instruction below

### Linux

Untested in our lab, follow: [https://gpufit.readthedocs.io/en/latest/installation.html](https://gpufit.readthedocs.io/en/latest/installation.html)


# Github/Gitlab details for GPUFit

- add details of Gitlab/Github, also info on Gpufit stuff
- we should probably have our own README_QSL.md file in the gpufit repo
- Most important Gpufit details so I don't forget:
	- 2 remotes: origin (Github, Gpufit) and cloud (Gitlab, QSL)
	- a few branches: 
		- master (don't touch this, unless rebasing from it -> this is their code)
		- master_QSL (touch this a little bit, 'our' (QSL) version of master)
		- any other branches you want to use (e.g. per feature/per user/per computer etc.)
	- so plan is to keep track of origin(-al) repo via rebasing, if they add new features,
	  but have our own remote (cloud, on gitlab), that we actually use as a centralised code store
	  for our group & all of our added fitmodels etc.
	- NEVER push to origin/github (probably not allowed anyway)


# Gpufit install

### Methodology:
 - check card details: what is it's compute capability? (nvidea website)
    - what cuda toolkits version to you need? (wikipedia calls this 'SDK')
 - check compatibility of cuda toolkit with OS version.
 - check compatibility with gpu driver.
 - check compatibility with visual studio.
 - yes you do need overlap between all of these things. Best to work it out before you start.
    - if you get it wrong, you'll need to uninstall everything and start again.

### Where to find these details:

 - is card cuda-compatible? also, what is the compute capability (roughly the cuda generation/micro-architecture) of the graphics card?
    - nvidea has a website for this, [link](https://developer.nvidia.com/cuda-gpus)
 - what cuda toolkits version can I use with my card?
    - most useful resource for me has been the cuda wiki page [link](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
    - 'SDK' is equivalent to 'cuda toolkits version'
    - this page also has a table you can use to determine the compute capability of your graphics card
 - cuda toolkits archive [link](https://developer.nvidia.com/cuda-toolkit-archive)
    - also links to documentation
    - section: 'installation guide windows' gives info on what OS version and visual studio version will work with the selected toolkit version
 - to find out about your nvidea driver:
    - in windows make sure you update it first (through device manager > display adapters)
    - also probably best to ensure the card isn't being used for graphics output!
    - right click on desktop > nvidea control panel > system information (bottom right), will display driver version
 - to find driver/cuda toolkits compatibility:
    - see nvidia cuda compatibility docs, I got this link at top of google [link](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
    - go to section 5.1: Support > Hardware Support
    - will show a table: compatibility between hardware generation (Fermi, Kepler etc.), compute capability and graphics driver version
        - this is an important one!!!


### Installation Procedure

 - Install microsoft visual studio (VS)
    - NOT TO BE CONFUSED WITH VISUAL STUDIO CODE
    - best to go with an older rather than newer version for compatibility with everything above
    - note 'community' version is free
    - where to get: [link](https://visualstudio.microsoft.com/vs/older-downloads/)
        - for older versions need a live account, I believe
    - pc will probably want to restart

 - install cuda toolkits

 - can then install [cmake](https://cmake.org/)
    - ensure it has been added to path

 - useful to also install BOOST C++ library [link](https://www.boost.org/)
    - will take a little while to build itself (run bootstrap file)
    - this allows you to run tests after building gpufit source

 - install python
    - I used windows msi install (>3.8)
    - ensure you add to path!
    - ensure pip installed!
    - make a folder at root (or C:) called 'src' -> will put all code here

 - environment management:
    - either conda (miniconda) or pipenv
    - miniconda install:
        - [link](https://docs.conda.io/en/latest/miniconda.html)
    - pipenv install:
        - `pip install --user pipx`
        - `pipx install pipenv`

 - install git
    - [git install link](gitforwindows.org)
    - connect to gitlab -> follow notes in version control section
    - connecting git to sublime merge (in particular for gitlab)
        - just ensure you are using ssh links not https!



### Building from source

- Grab latest QSL fork of gpufit from gitlab [link](https://gitlab.unimelb.edu.au/QSL/gpufit)

- Can basically follow instruction in gpufit docs [link](https://gpufit.readthedocs.io/)
    - In general, the gpufit docs are quite good, but you need to fiddle around quite a bit!

- compiler configuration (Cmake):
    -  First, identify the directory which contains the Gpufit source code (for example, on a Windows computer the Gpufit source code may be stored in C:\src\gpufit). Next, create a build directory outside the source code source directory (e.g. C:\src\gpufit-build). Finally, run cmake to configure and generate the compiler input files. The following commands, executed from the command prompt, assume that the cmake executable (e.g. C:\Program Files\CMake\bin\cmake.exe) is automatically found via the PATH environment variable (if not, the full path to cmake.exe must be specified). This example also assumes that the source and build directories have been set up as specified above.
        - `cd C:\src\gpufit-build`
        - `cmake -G "Visual Studio 12 2013 Win64" C:\Sources\Gpufit`
    - I then open up the cmake gui (which will auto-populate fields from this previous cmake run) to edit some more things:
        - set \_USE_CBLAS flag to be true
        - add BOOST_ROOT variable to wherever you installed BOOST

- compiling (visual studio)
    - After configuring and generating the solution files using CMake, go to the desired build directory and open Gpufit.sln using Visual Studio. Select the “Debug” or “Release” build options, as appropriate. Select the build target “ALL_BUILD”, and build this target. If the build process completes without errors, the Gpufit binary files will be created in the corresponding “Debug” or “Release” folders in the build directory.
    - The unit tests can be executed by building the target “RUN_TESTS” or by starting the created executables in the output directory from the command line. (I RECOMMEND YOU RUN SOME TESTS!)

- Building python wheel file
    - uninstall any previous version you installed (`pip uninstall pygpufit`)
    - `pip install C:\src\gpufit-build-Release\pyGpufit\dist\wheel_file_here.wh`

- Haven't tested building on linux, but it's probably easier :)


# Documentation

Follow docstring pattern in files, e.g.:

- Docstrings for module variables:
```python
module_variable = 1
"""Docstring for module_variable."""
```

- Docstrings for class variables and methods:
```python
class C:
    class_variable = 2
    """Docstring for class_variable."""

    def __init__(self):
        self.variable = 3
        """Docstring for instance variable."""
```

- Docstrings for each module (backticks convert to links in html):
```python
# -*- coding: utf-8 -*-
"""
This module defines...

Classes
-------
 - `QDMPy.<this sub-package name>.<this module name (filename)>.ClassNameOne`

Functions
---------
 - `QDMPy.<this sub-package name>.<this module name (filename)>.Function1`
"""
```

- Docstrings for all functions (and methods etc.) describing inputs and outputs:
```python
def fit_ROI_avg(options, sig_norm, sweep_list, fit_model):
    """
    Fit the average of the measurement over the region of interest specified.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_list, y, x].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `QDMPy.fit._models.FitModel` object.

    Returns
    -------
    `QDMPy.fit._shared.ROIAvgFitResult` object containing the fit result (see class specifics)
    """
```

To build documentation (html) using [pdoc](https://pdoc3.github.io/pdoc/doc/pdoc/#pdoc) tool:
- navigate to dir: QDMPy_git/QDMPy, e.g. for Sam: ~/src/nv/QDMPy_proj/QDMPy_git/QDMPy
- cmd: `pdoc --html --force --output-dir doc .`
- the period at the end of the command above is required!
