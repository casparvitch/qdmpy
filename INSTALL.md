Install Instructions for qdmpy
==============================

# Usage

- follow instructions below for install
- copy one of the example notebooks/scipts to another directory
- open the directory in jupyterlab from a shell: `jupyter-lab`
- edit the options, see '/src/systems/unimelb_defaults.json' for rough docs

# Examples/tests

- grab testing datasets from Sam for now, will upload somewhere shared in future.

# Basic install
(Assuming you know git - see below)

- [clone](https://gitlab.unimelb.edu.au/sscholten/qdmpy)
- install python3.8
- navigate to root-dir (containing 'src' etc.)
```bash
pip install . # will install deps
pip install jupyterlab
```

## If above install didn't work
- install conda/miniconda
- this is useful as it allows us to specify the python version

```bash
conda info # verify install 
conda update -n base conda # update base conda
conda create --name QDMPy python=3.8 # new environment called QDMPy
conda info --envs # list environments
conda activate QDMPy # activate environment
conda install numpy # installing dependencies
conda install -c conda-force matplotlib-scalebar # some deps are in conda-forge
```
- navigate to root-dir (containing 'src' etc.), [info](https://github.com/conda/conda-build/issues/4251)
```bash
pip install --no-build-isolation --no-deps -e .
```
- you will need to install your own deps:

```bash
conda install numpy "matplotlib>=3.4.0" "scipy>=1.7" numba tqdm psutil simplejson pandas dill astropy
conda install -c conda-forge "matplotlib-scalebar>=0.7.2" pyfftw pysimplegui
pip install rebin foronoi python-polylabel # hopefully these don't muck anything up, I will be trying to remove these soon
```

- although the above may have issues, see e.g. [here](https://github.com/conda/conda-build/issues/4251)

# Documentation

- navigate to root-dir (containing 'src' etc.)
```bash
conda install -c conda-forge pdoc3
pdoc3 --output-dir docs --html --template-dir ./docs/ --force ./src/qdmpy # or eq. on windows
```

# prospector

- navigate to root-dir (containing 'src' etc.)
```bash 
pip install prospector # or use conda...
prospector --profile qdmpy.prospector.yaml -o grouped:prospector.log
```

# jupyter-lab widgets

- Install [nodejs](https://nodejs.org/en/) and [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension).

- For help, see [here](https://stackoverflow.com/questions/49542417/how-to-get-ipywidgets-working-in-jupyter-lab).  

- To save widget state: in jupyterlab > settings > advanced settings editor > jupyter widgets > saveState: true

# Version control / git

The project is housed on [Gitlab](https://gitlab.unimelb.edu.au/sscholten/qdmpy), you will need to be given access by an owner and sign in with uni credentials. To communicate with the Gitlab server you will need to setup an ssh key (on each device connected, there will need to be one on each lab computer as well). My installation instructions below are taken from the Gitlab Docs [here](https://docs.gitlab.com/ee/ssh/).

You can also use Gitlab in-browser, i.e. not using the git commands at all. This is not recommended but can be great in a pinch.

Tip: It's a lot easier to read diffs/merges using the side-by-side visual. Somewhere in your diff tool/online it will have this options, give it a shot.


## git installation

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

# Gpufit install

Can install a binary from [here](https://github.com/gpufit/Gpufit/releases).

However this does not include the model functions we require (e.g. lorentzians, exponential decays) so you will need to write these into the source and rebuild it (each time you want a new model function...). Details below.

## Linux

Follow [instructions](https://gpufit.readthedocs.io/en/latest/installation.html) - worked easier than Windows.


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

## Install methodology:
 - check card details: what is it's compute capability? (nvidea website)
    - what cuda toolkits version to you need? (wikipedia calls this 'SDK')
 - check compatibility of cuda toolkit with OS version.
 - check compatibility with gpu driver.
 - check compatibility with visual studio.
 - yes you do need overlap between all of these things. Best to work it out before you start.
    - if you get it wrong, you'll need to uninstall everything and start again.

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
        - set \_USE_CBLAS flag to be true (if you get errors when building try False -> sometimes gpufit gets the name of the cuBLAS dll incorrect). Still not sure how to fix this one, not a big issue though.
        - add BOOST_ROOT variable to wherever you installed/unpacked BOOST

- compiling (visual studio)
    - After configuring and generating the solution files using CMake, go to the desired build directory and open Gpufit.sln using Visual Studio. Select the “Debug” or “Release” build options, as appropriate. Select the build target “ALL_BUILD”, and build this target. If the build process completes without errors, the Gpufit binary files will be created in the corresponding “Debug” or “Release” folders in the build directory.
    - The unit tests can be executed by building the target “RUN_TESTS” or by starting the created executables in the output directory from the command line. (I RECOMMEND YOU RUN SOME TESTS!)

- Building python wheel file
    - ENSURE wheel installed (pip install wheel) BEFORE compilation
    - uninstall any previous version you installed (`pip uninstall pygpufit`)
    - `pip install C:\src\gpufit-build-Release\pyGpufit\dist\wheel_file_here.wh`


