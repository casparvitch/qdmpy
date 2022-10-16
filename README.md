<!-- Required extensions: pymdownx.betterem, pymdownx.tilde, pymdownx.emoji, pymdownx.tasklist, pymdownx.superfences -->
<!-- Don't edit these comments: 'Required extensions' is for ReText editing, 'MarkdownTOC' for sublime text  auto-generated table of contents-->

Quantum Diamond MicroscoPy
==========================

# Table of Contents

<!-- MarkdownTOC autolink="true" bracket="round" autoanchor="true" -->

- [Introduction](#introduction)
- [Usage](#usage)
    - [Module Hierarchy](#module-hierarchy)
    - [Order of operations](#order-of-operations)
    - [Ok but how do I actually use qdmpy Sam?](#ok-but-how-do-i-actually-use-qdmpy-sam)
    - [Magsim usage](#magsim-usage)
- [Updating your qdmpy version](#updating-your-qdmpy-version)
- [Installation](#installation)
    - [Installing with pip on new system](#installing-with-pip-on-new-system)
    - [Jupyterlab installation](#jupyterlab-installation)
    - [Saving widget state](#saving-widget-state)
    - [Exporting notebook to pdf](#exporting-notebook-to-pdf)
- [Version control](#version-control)
    - [VC installation](#vc-installation)
- [Environment management with conda](#environment-management-with-conda)
    - [Quickstart](#quickstart)
    - [Packages](#packages)
- [Gpufit install](#gpufit-install)
    - [Linux](#linux)
    - [Github/Gitlab details for GPUFit](#githubgitlab-details-for-gpufit)
    - [Install methodology:](#install-methodology)
    - [Where to find these details:](#where-to-find-these-details)
    - [Installation Procedure](#installation-procedure)
    - [Building GPUFit from source](#building-gpufit-from-source)
- [Documentation conventions](#documentation-conventions)
- [Options reference](#options-reference)
- [How to implement your own custom System class](#how-to-implement-your-own-custom-system-class)

<!-- /MarkdownTOC -->


<a id="introduction"></a>
# Introduction

If you're installing for the first time scroll down to 'Installing with pip on new system'.

<a id="usage"></a>
# Usage

Best used in a jupyter notebook to avoid producing a million graphs. In future a command-line suitable hook will be defined.

Usage is best understood through the example notebooks.


<a id="module-hierarchy"></a>
## Module Hierarchy

View in text editor if scrambled.

```
      =================================
      Subpackage dependency graph (DAG)
      =================================

                                                           +--------------------------+
                   +-----+                                 |  ===                     |
                   |qdmpy|                                 |  Key                     |
   +---------------+=====+------------+                    |  ===                     |
   |               +-----+            |                    |                          |
   |                                  |                    | +----+                   |
   v                                  v                    | |name|  =  Package       |
+--+---+  +--+  +-----+  +------+  +--+-+  +------+        | |====|                   |
|system|  |pl|  |field|  |source|  |plot|  |magsim|        | +----+                   |
|======|  |==|  |=====|  |======|  |====|  |======|        |                          |
+--+---+  +-++  +--+--+  +--+---+  +-+--+  +--+---+        |  name   =  Module        |
   |        |      |        |        |        |            |  +--+                    |
   |        |      |        |        |        |            |                          |
   |        |      |        |        |        |            |                          |
   |        v      v        v        |        |            |  +-->   =  Dependency    |
   |     +--+------+--------+------+<+        |            |                          |
   +---->+          shared         |          |            +--------------------------+
         |          ======         +<---------+
         | geom                    |
         | +--+    misc    itool   |
         |         +--+    +-+-+   |
         |                   |     |             CANNOT IMPORT FROM HIGHER IN HEIRARCHY
         | fourier           v     |
         | +-----+        polygon  |
         |                +--+--+  |
         | linecut           |     |
         | +-+---+           |     |
         |   |               |     |
         |   +---+--------+--+     |
         |       |        |        |
         |       v        v        |
         |    widget    json2dict  |
         |    +----+    +-------+  |
         +-------------------------+

```

<a id="order-of-operations"></a>
## Order of operations

View in text editor if scrambled.

```
+---------------------------------------------------------------------------------------+                                                                                 
| Methods are qdmpy.{what is listed here}                                               |                                                                                 
|---------------------------------------------------------------------------------------|                                                                                 
| Functions                         Variables: type          Plotting                   |                                                                                 
|---------------------------------------------------------------------------------------|                                                                                 
|                                   options = {                                         |                                                                                 
| initialize                           ...                                              |                                                                                 
|                                   }: dict or json                                     |                                                                                
|                                   (same for ref)                                      |                                                                                 
|                                                                                       |                                                                                 
| pl.load_image_and_sweep           sig_norm: 3D ndarray     plot.roi_pl_image          |                                                                                 
| pl.reshape_dataset                sweep_list: list         plot.aoi_pl_image          |                                                                                 
|                                                            plot.aoi_spectra           |                                                                                 
|                                                                                       |                                                                                 
| pl.save_pl_data                                                                       |                                                                                 
|                                   ref_fit_params: dict                                |                                                                                 
| pl.load_ref_exp_pl_fit_results    ref_sigmas: dict                                    |                                                                                 
|                                                                                       |                                                                                 
| pl.define_fit_model               fit_model: FitModel                                 |                                                                                 
| pl.fit_roi_avg_pl                                          plot.roi_avg_fits          |                                                                                 
|                                                            plot.aoi_spectra_fit       |                                                                                 
|                                                                                       |                                                                                 
| pl.get_pl_fit_result              pixel_fit_params: dict   plot.pl_param_images       |                                                                                 
|                                   sigmas: dict             plot.pl_param_sigmas       |                                                                                 
| pl.save_pl_fit_results                                     plot.pl_params_flattened   |                                                                                 
| pl.save_pl_fit_sigmas                                                                 |                                                                                 
|                                                                                       |                                                                                 
| field.odmr_field_retrieval        field_res: dict                                     |                                                                                 
|                                                            plot.bnvs_and_dshifts      |                                                                                 
| field.add_bfield_theta_phi                                 plot.bfield                |                                                                                 
| field.save_field_calcs                                     plot.dshift_fit            |                                                                                 
|                                                            plot.field_param_flattened |                                                                                 
|                                                            plot.bfield_consistency    |                                                                                 
|                                                            plot.bfield_theta_phi      |                                                                                 
|                                                                                       |                                                                                 
|                                                                                       |                                                                                 
| source.odmr_source_retrieval      source_params: dict                                 |                                                                                 
| source.save_source_params                                  plot.current               |                                                                                 
|                                                            plot.current_stream        |                                                                                 
|                                                            plot.divperp_j             |                                                                                 
|                                                                                       |                                                                                 
| save_options                                               plot.magnetization         |                                                                                 
|                                                                                       |                                                                                 
|                                                            plot.other_measurements    |                                                                                 
+---------------------------------------------------------------------------------------+
```

<a id="ok-but-how-do-i-actually-use-qdmpy-sam"></a>
## Ok but how do I actually use qdmpy Sam?

Look at the example notebooks. Magsim and other parts of the code are a bit more separate and don't use notebooks, but they have their own example scripts - lovely.

How to run an example script (if you're in the example directory in a terminal, which you can navigate to with the commands `cd` and `dir` on windows):

```Bash
python <script_name>.py
```

How to load jupyter notebooks, run this (e.g. from the examples dir):
```Bash
jupyter-lab
```

... then click on the relevant notebook in your browser. I hope I don't need to spell it out much more :)

<a id="magsim-usage"></a>
## Magsim usage

TODO. Code still in state of flux.

<a id="updating-your-qdmpy-version"></a>
# Updating your qdmpy version

- first get the new code:
    - open sublime merge
    - if you aren't already, checkout the main branch: on LHS double click on 'main' under BRANCHES
    - in top right click down arrow (not down triangle) for 'pull'. 
    - Alternatively in top-middle there's a bar that says 'main', (left of magnifying glass symbol), click on that, delete whatever is there (probably 'checkout branch'), type 'pull', hit enter twice. (this top-middle bar is for sending commands directly to git instead of via the GUI).
    - if you have a 'local_<system>' branch or similar, check it out (double left click on LHS), then right click on 'main' branch on LHS and choose 'merge main into local'. Now you have the latest code in your own branch that's quarantined from causing problems.

- now install new version:
    - open up terminal (anaconda prompt or whatever you use)
    - test that you can access the current install:
        - conda activate qdmpy (or whatever environment you use)
        - python (or python3 on some systems)
        - `import qdmpy` (all g to continue if this works)
    - navigate to qdmpy folder (will contain folders 'src', 'docs', 'examples' etc.)
        - for you probably in C:\src\qdmpy or something like that
        - see where you are with `dir` (windows cmd) or `ls` (unix/bash)
        - navigate to new dir with `cd`, e.g. `cd C:\src\qdmpy` or if you're currently in C: `cd src\qdmpy` etc.
    - build package: `python setup.py bdist_wheel` (this should run without error)
    - navigate to distribution folder: `cd dist`
    - install package: `pip install qdmpy-<VERSION_NUMBER>-[...].whl` (whatever the wheel .whl file with the latest version is)
    - navigate elsewhere, e.g. `cd C:\`
    - test that it's installed properly: `python` then `import qdmpy`.



<a id="installation"></a>
# Installation


<a id="installing-with-pip-on-new-system"></a>
## Installing with pip on new system

You're installing for the first time etc. and want to be able to run 'import qdmpy'. Follow these instructions.

Download source from gitlab, and go to the root directory (containing 'src', 'docs' etc.). Run `python3 setup.py bdist_wheel` if there's no wheel file (probably contained within 'dist' directory, a '.whl' file). Once you have the wheel file (note I haven't tested on different platforms...) run `pip3 install <PATH_TO_WHEEL_FILE>.whl`. Note on windows those commands should probably be python and pip, not python3 and pip3.

On linux, to quickly uninstall, build new wheel, install and then run jupyterlab (change version number to current):
```bash
pip3 uninstall qdmpy -y && python3 setup.py bdist_wheel && pip3 install ./dist/qdmpy-1.1.0-py3-none-any.whl && jupyter-lab  
```


<a id="jupyterlab-installation"></a>
## Jupyterlab installation

Install [nodejs](https://nodejs.org/en/) and [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension).

For help, see [here](https://stackoverflow.com/questions/49542417/how-to-get-ipywidgets-working-in-jupyter-lab).

<a id="saving-widget-state"></a>
## Saving widget state

In jupyterlab > settings > advanced settings editor > jupyter widgets > saveState: true

<a id="exporting-notebook-to-pdf"></a>
## Exporting notebook to pdf

- install [pandoc](https://pandoc.org/installing.html)
- need to change to %matplotlib inline?
	- check Github issues (e.g. [#16](https://github.com/matplotlib/ipympl/issues/16), [#150](https://github.com/matplotlib/ipympl/issues/150) and [#176](https://github.com/matplotlib/ipympl/pull/176))

<a id="version-control"></a>
# Version control

The project is housed on [Gitlab](https://gitlab.unimelb.edu.au/sscholten/qdmpy), you will need to be given access by an owner and sign in with uni credentials. To communicate with the Gitlab server you will need to setup an ssh key (on each device connected, there will need to be one on each lab computer as well). My installation instructions below are taken from the Gitlab Docs [here](https://docs.gitlab.com/ee/ssh/).

You can also use Gitlab in-browser, i.e. not using the git commands at all. This is not recommended but can be great in a pinch.

Tip: It's a lot easier to read diffs/merges using the side-by-side visual. Somewhere in your diff tool/online it will have this options, give it a shot.


<a id="vc-installation"></a>
## VC installation


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
cat ~/.ssh/id_ed25519.pub | clip
```

macOS:

```Bash
pbcopy < ~/.ssh/id_ed25519.pub
```

WSL/GNU/Linux (you may need to install xclip but it will give you instructions):

```Bash
xclip -sel clip < ~/.ssh/id_ed25519.pub
```

Now we can check whether that worked correctly:

```Bash
ssh -T git@git.unimelb.edu.au
```

If it's the first time you connect to Gitlab via SSH, you will be asked to verify it's authenticity (respond yes). You may also need to provide your password. If it responds with *Welcome to Gitlab, @username!* then you're all setup.


<a id="environment-management-with-conda"></a>
# Environment management with conda

Conda/Anaconda is a software package that allows for multiple 'environments' i.e. collections of packages, even different versions of Python, on the same machine. It also allows us to share our environment on our personal machines between people so we can all have a common 'background' to run the code on - very useful for avoiding errors!

There are various methods for installing conda, but PyCharm/Atom/most IDEs come with it. If you like working from the command line, download [Miniconda](https://docs.conda.io/en/latest/miniconda.html). I find this miniconda easier as its a bit clearer what's going on at all times.

<a id="quickstart"></a>
## Quickstart

Download miniconda from anaconda site. Open an anaconda prompt (if on windows), then verify install and update with

```Bash
conda info
conda update -n base conda
```

To create an environment use (currently must be 3.8, due to pyfftw constraints...)

```Bash
conda create --name ENVNAME python=3.8
````

To activate the environment (actually use it in the terminal)

```Bash
conda activate ENVNAME
```

To list the environments on this machine:

```Bash
conda info --envs
```

[Cheatsheet](https://docs.anaconda.com/anaconda/user-guide/cheatsheet/) tells you literally everything you need to know.

<a id="packages"></a>
## Packages

Then you can specify, separately from the rest of your pc, the packages you want to be installed/used when running scripts when this environment is activated. To search for packages go to anaconda.org and just type in the name you want. NB: Try not to use pip inside conda for installation of external packages. NB2: The install process for qdmpy *should* automatically install all required dependencies... I hope.


```Bash
conda install numpy
```

If that doesn't work conda-forge usually will


```Bash
conda install -c conda-forge numpy
```



<a id="gpufit-install"></a>
# Gpufit install

Can install a binary from [here](https://github.com/gpufit/Gpufit/releases).

However this does not include the model functions we require (e.g. lorentzians, exponential decays) so you will need to write these into the source and rebuild it (each time you want a new model function...). Details below.

<a id="linux"></a>
## Linux

Untested in our lab, follow: [https://gpufit.readthedocs.io/en/latest/installation.html](https://gpufit.readthedocs.io/en/latest/installation.html)


<a id="githubgitlab-details-for-gpufit"></a>
## Github/Gitlab details for GPUFit

- Our fork of GPUFit (with our lorentzian models etc.) is held [here](https://gitlab.unimelb.edu.au/QSL/gpufit)
- Most important Gpufit details:
	- 2 remotes: origin (Github, Gpufit i.e. **THEM**) and cloud (Gitlab, QSL i.e. **US**)
	- a few branches: 
		- master (don't touch this, unless rebasing from it -> this is **their** code / branch)
		- master_QSL (touch this a little bit, '**our**' (QSL) version of master)
		- any other branches you want to use (e.g. per feature/per user/per computer etc.)
	- so plan is to keep track of origin(-al) repo via rebasing (/ merging), if they add new features,
	  but have our own remote (cloud, on gitlab), that we actually use as a centralised code store
	  for our group & all of our added fitmodels etc.
	- NEVER push to origin/github (probably not allowed anyway)
- so basically all you need to do: 
    - don't get code from github, get it from gitlab (**US**)
    - switch to master_QSL branch
    - install (give yourself a couple of hours or ask me for help)

<a id="install-methodology"></a>
## Install methodology:
 - check card details: what is it's compute capability? (nvidea website)
    - what cuda toolkits version to you need? (wikipedia calls this 'SDK')
 - check compatibility of cuda toolkit with OS version.
 - check compatibility with gpu driver.
 - check compatibility with visual studio.
 - yes you do need overlap between all of these things. Best to work it out before you start.
    - if you get it wrong, you'll need to uninstall everything and start again.

<a id="where-to-find-these-details"></a>
## Where to find these details:

 - is card cuda-compatible? also, what is the compute capability (roughly the cuda generation/micro-architecture) of the graphics card?
    - nvidea has a website for this, [link](https://developer.nvidia.com/cuda-gpus)
 - what cuda toolkits version can I use with my card/driver version?
    - most useful resource for me has been the cuda wiki page [link](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
    - 'SDK' is equivalent to 'cuda toolkits version'
    - this page also has a table you can use to determine the compute capability of your graphics card
    - note gpufit was tested by the authors from 6.5-10.1 (but latest should be fine)
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


<a id="installation-procedure"></a>
## Installation Procedure

 - Install/update gpu driver

 - Install microsoft visual studio (MSVS)
    - *not to be confused with Visual Studio Code*
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
    - either conda (miniconda) or pipenv (be nice on yourself and use conda)
    - miniconda install:
        - [link](https://docs.conda.io/en/latest/miniconda.html)
    - pipenv install:
        - `pip install --user pipx`
        - `pipx install pipenv`

 - install git
    - [git install link](https://gitforwindows.org)
    - connect to gitlab -> follow notes in version control section
    - connecting git to sublime merge (in particular for gitlab)
        - **ensure you are using ssh links not https!**



<a id="building-gpufit-from-source"></a>
## Building GPUFit from source
 
- Grab latest QSL fork of gpufit from gitlab [link](https://gitlab.unimelb.edu.au/QSL/gpufit)
    - **MAKE SURE** you're on the master_qsl branch!!! -> this branch has our additions/fixes!

- Can basically follow instruction in gpufit docs [link](https://gpufit.readthedocs.io/)
    - In general, the gpufit docs are quite good, but you need to fiddle around quite a bit!

- compiler configuration (Cmake):
    -  First, identify the directory which contains the Gpufit source code (for example, on a Windows computer the Gpufit source code may be stored in C:\src\gpufit). Next, create a build directory outside the source code source directory (e.g. C:\src\gpufit-build). Finally, run cmake to configure and generate the compiler input files. The following commands, executed from the command prompt, assume that the cmake executable (e.g. C:\Program Files\CMake\bin\cmake.exe) is automatically found via the PATH environment variable (if not, the full path to cmake.exe must be specified). This example also assumes that the source and build directories have been set up as specified above.
        - `cd C:\src\gpufit-build`
        - `cmake -G "Visual Studio 12 2013 Win64" C:\Sources\Gpufit`
    - I then open up the cmake gui (which will auto-populate fields from this previous cmake run) to edit some more things:
        - set \_USE_CBLAS flag to be true (if you get errors when building try False -> sometimes gpufit gets the name of the cuBLAS dll incorrect)
        - add BOOST_ROOT variable to wherever you installed/unpacked BOOST

- compiling (visual studio)
    - After configuring and generating the solution files using CMake, go to the desired build directory and open Gpufit.sln using Visual Studio. Select the “Debug” or “Release” build options, as appropriate. Select the build target “ALL_BUILD”, and build this target. If the build process completes without errors, the Gpufit binary files will be created in the corresponding “Debug” or “Release” folders in the build directory.
    - The unit tests can be executed by building the target “RUN_TESTS” or by starting the created executables in the output directory from the command line. (I RECOMMEND YOU RUN SOME TESTS!)

- Building python wheel file
    - ENSURE wheel installed (pip install wheel) BEFORE compilation
    - uninstall any previous version you installed (`pip uninstall pygpufit`)
    - `pip install C:\src\gpufit-build-Release\pyGpufit\dist\wheel_file_here.wh`

- Haven't tested building on linux, but it's probably easier :)


<a id="documentation-conventions"></a>
# Documentation conventions

Follow (numpy) docstring pattern in files, e.g.:

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
 - `qdmpy.<this sub-package name>.<this module name (filename)>.ClassNameOne`

Functions
---------
 - `qdmpy.<this sub-package name>.<this module name (filename)>.Function1`
"""
```

- Docstrings for all functions (and methods etc.) describing inputs and outputs:
```python
def fit_roi_avg_pl(options, sig_norm, sweep_list, fit_model):
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
    fit_model : `qdmpy.pl.model.FitModel` object.

    Returns
    -------
    `qdmpy.pl.common.ROIAvgFitResult` object containing the fit result (see class specifics)
    """
```

To build documentation (html) using [pdoc](https://pdoc3.github.io/pdoc/doc/pdoc/#pdoc) tool:
- navigate to the docs directory and read the README.md file over there!


<a id="options-reference"></a>
# Options reference

Maintaining two references for the available options didn't go so well. Check the unimelb_defaults.json file (in the system folder).

<a id="how-to-implement-your-own-custom-system-class"></a>
# How to implement your own custom System class

Largely TODO at the moment. Key things to check:
- the '_SYSTEMS' variable in systems.py
- implement your own child of the System class, including all its methods (unless you're inheriting from the unimelb defaults, which would be the case if you're at RMIT/Unimelb).