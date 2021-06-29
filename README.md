Quantum Diamond MicroscoPy
==========================

- [Introduction](#introduction)
- [Usage](#usage)
            - [Flow diagram](#flow-diagram)
- [Installation](#installation)
        - [Installing with pip on new system](#installing-with-pip-on-new-system)
        - [Re-building wheel file](#re-building-wheel-file)
        - [Jupyterlab installation](#jupyterlab-installation)
        - [Saving widget state](#saving-widget-state)
        - [Exporting notebook to pdf](#exporting-notebook-to-pdf)
        - [Version control](#version-control)
            - [VC installation](#vc-installation)
        - [Environment management \(conda/pipenv\)](#environment-management-condapipenv)
        - [Gpufit](#gpufit)
        - [Linux](#linux)
- [Github/Gitlab details for GPUFit](#githubgitlab-details-for-gpufit)
- [Gpufit install](#gpufit-install)
        - [Methodology:](#methodology)
        - [Where to find these details:](#where-to-find-these-details)
        - [Installation Procedure](#installation-procedure)
        - [Building from source](#building-from-source)
- [Documentation conventions](#documentation-conventions)


# Introduction

If you're installing for the first time scroll down to 'Installing with pip on new system'.


# Usage

Best used in a jupyter notebook to avoid producing a million graphs. In future a command-line suitable hook will be defined.

Usage is best understood through the example notebooks.

#### Flow diagram

View in text editor if scrambled.

```
---------------------------------------------------------------------------------------------------
|                                                                                                 |
|    plotting                  processing           ┌──────────────┐                              |
|    xxxxxxxx                  xxxxxxxxxx           │              │                              |
|                                                   │ options.json │                              |
|     saving                                        │     or       │       ┌─────────────────┐    |
|     xxxxxx                                        │ options = {  │       │ ref_options     │    |
|                                                   │     ...      │       │                 │    |
|                                                   │ }            │       │ ref_options_dir │    |
|                           ┌──────────────┐        └──────┬───────┘       └───────┬─────────┘    |
|                           │              │               │                       │              |
|                           │ Raw pl data  │           load_options                │              |
| set_mpl_rcparams          │    on disk   │               │                       │              |
|                           │              │           ┌───┴─────┐                 │              |
|                           └─────┬────────┘           │         │            ┌────┴───────┐      |
|                                 │                    │ Options ├───────┐    │Ref Options │      |
|                           load_image_and_sweep ──────┤ dict    │       │    │Dict        │      |
|                           reshape_dataset            │         │       │    └────┬───────┘      |
|                                 │                    └───┬─────┘       │         │              |
| plot_ROI_pl_image          ┌────┴───────┐                │             │         │              |
| plot_AOI_pl_images         │ sig_norm   │                │             │         │              |
| plot_AOI_spectra           │ sweep_list │          define_fit_model    │         │              |
|                            └────┬───────┘                │             │         │              |
| save_pl_data                    │                    ┌───┴──────┐      │         │              |
|                           get_pl_fit_result ─────────┤fit_model │      │         │              |
|                                 │                    └──────────┘      │         │              |
| fit_roi_avg_pl                     │                                      ├──load_reference_exp.   |
| plot_ROI_avg_fits        ┌──────┴───────────┐                          │      fit_results       |
| fit_aois_pl                 │ pixel_fit_params │                          │         │              |
| plot_AOI_spectra_fit     │                  ├────────────┬─────────────┘         │              |
| plot_param_images        │   ref_sigmas     │            │                 ┌─────┴─────────┐    |
| plot_params_flattened    └──────┬───────────┘            │                 │ ref_fit_params│    |
|                                 │                        │                 │ ref_sigmas    │    |
|                                 │                        │                 └─────┬─────────┘    |
|                           odmr_field_retrieval───┐       x                       │              |
|                                 │                └───────────────────────────────┘              |
| save_field_calcs           ┌────┴─────────┐              x                                      |
| plot_bnvs_and_dshifts      │ bnvs         │              │                                      |
| plot_bfield                │ dshifts      │              │                                      |
| plot_dshift_fit            │ field_params │              │                                      |
| plot_field_param_flattened │ sigma_params │              │                                      |
|                            └──────────────┘          save_options                               |
|                                                                                                 |
---------------------------------------------------------------------------------------------------
```

# Installation


### Installing with pip on new system

You're installing for the first time etc. and want to be able to run 'import qdmpy'. Follow these instructions.

Download source from gitlab, and go to the root directory (containing 'src', 'docs' etc.). Run `python3 setup.py bdist_wheel` if there's no wheel file (probably contained within 'dist' directory, a '.whl' file). Once you have the wheel file (note I haven't tested on different platforms...) run `pip3 install <PATH_TO_WHEEL_FILE>.whl`. Note on windows those commands should probably be python and pip, not python3 and pip3.

On linux, to quickly uninstall, build new wheel, install and then run jupyterlab:
```bash
pip3 uninstall qdmpy -y && python3 setup.py bdist_wheel && pip3 install ./dist/qdmpy-0.1.0-py3-none-any.whl && jupyter-lab  
```


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

### Version control

The project is housed on [Gitlab](https://gitlab.unimelb.edu.au/sscholten/qdmpy), you will need to be given access by an owner and sign in with uni credentials. To communicate with the Gitlab server you will need to setup an ssh key (on each device connected, there will need to be one on each lab computer as well). My installation instructions below are taken from the Gitlab Docs [here](https://docs.gitlab.com/ee/ssh/).

You can also use Gitlab in-browser, i.e. not using the git commands at all. This is not recommended but can be great in a pinch.

Tip: It's a lot easier to read diffs/merges using the side-by-side visual. Somewhere in your diff tool/online it will have this options, give it a shot.


#### VC installation


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


### Environment management (conda/pipenv)


TODO -> conda best for windows, so probably go with that. With new install practises (i.e. from wheel file) this won't be necessary as they should be bundled. Note: if installing from wheel file then user cannot change the source files? This is kind of the point of installing properly. What would user want to change? Perhaps constants etc. but then we would want them to build
the wheel again.


### Gpufit

Can install a binary from [here](https://github.com/gpufit/Gpufit/releases).

However this does not include the model functions we require (e.g. lorentzians, exponential decays) so you will need to write these into the source and rebuild it (each time you want a new model function...). Details below.

- [ ] Goal: keep our own branch with our model functions, rebase on master from time to time.
	- This way we can keep a group-wide collection of usable model functions
	- store in gitlab?
	- [multiple remotes, github](https://forum.sublimetext.com/t/working-with-multiple-remotes/53489/3)
- [ ] Keep in separate repo to qdmpy


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


### Installation Procedure

 - Install microsoft visual studio (MSVS)
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
    - MAKE SURE you're on the master_qsl branch!!! -> this branch has our additions/fixes!

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
    fit_model : `qdmpy.fit.model.FitModel` object.

    Returns
    -------
    `qdmpy.fit._shared.ROIAvgFitResult` object containing the fit result (see class specifics)
    """
```

To build documentation (html) using [pdoc](https://pdoc3.github.io/pdoc/doc/pdoc/#pdoc) tool:
- navigate to the root directory, e.g. ~/src/qdmpy_proj/qdmpy_git/ (which should contain the directory qdmpy)
- cmd: `pdoc --output-dir docs --html --config latex_math=True --force qdmpy`
- the period at the end of the command above is required!



# Options reference


### General parameters

- base_dir
    - Not required, if not "" it is prepended to filepath.
- filepath
    - Required, specify path to raw (pl) data.
- custom_output_dir_prefix
    - See below.
- custom_outpur_dir_suffix
    - See below.
- custom_output_dir
    - Any option key can be interpolated (to options[key]) with braces (like python f-strings) e.g. "ODMR_{fit_backend}" -> "ODMR_gpufit" if the fit backend was gpufit
    - Custom_output_dir must be an absolute path (E.g. from root/C:). Will override the path determined internally.
- ignore_ref
    - If true the reference (often no-MW) measurement is ignored
- additional_bins
    - Additional binning on dataset (local averaging). 0 does nothing (or 1). Must be multiple of 2 otherwise.
- system_name
    - Name of system used (see qdmpy.system)
- other_measurement_suffixes
    - List of suffix strings e.g. "\_T.txt" to look for, to plot non-NV measurements. E.g. for \_T.txt you might have stored T(t) during the experiment. They will be plot nicely for you. 
- remove_start_sweep
    - Removes data poitns from the start and end of each pixel's sweep data (e.g. freqs).
- remove_end_sweep
    - As above but for end of sweep
- normalisation
    - Style of the reference normalisation used. div = division and sub = subtraction

#### Microscope settings

- microscope_setup
    - Default options for the microscope. A dict containing 'sensor_pixel_size', 'default_objective_mag', 'default_objective_reference_focal_length' and 'default_camera_tube_lens'. Used to calculate pixel size.
- pixel_size
    - Provide pixel size manually.
- objective_mag
    - pixel_size = sensor_pixel_size * (focal length objective / focal length camera tube lens) where tube lens is the lens that focuses onto the camera, and: focal length objective = objective reference focal length / objective mag. Leave objective_mag and the below as None to use defaults (above). Otherwise these can be varied (e.g. commonly changing objective mag) and the code will handle it nicely.
- objective_reference_focal_length
- camera_tube_lens

#### Polygon settings

- polygon_nodes_path
    - Path to a polygon json file (containing nodes from a previous definition)
- mask_polygons_bground
    - Boolean, mask polygons for background calculation?
- annotate_polygons
    - Boolean, annotate polygons on all image plots?

#### region of interest (ROI) params

- ROI
    - For taking a region of interest and ignoring the rest of the FOV. Currently can be 'Full' or 'Rectangle'.
- ROI_start
    - Start of ROI (top left corner) if ROI: Rectangle.
- ROI_end
    - Bottom right corner of ROI rectangle.
- single_pixel_check
    - Pixel location used to check single pixel spectra + single pixel fit. 
- AOI_n_start
    - For some natural number 'n' (e.g. AOI_2_start) define a rectangular region to check (locally average) spectra and fit. Defines top left of region.
- AOI_n_end
    - Defines bottom right of region.

### Photoluminescence fitting parameters

- fit_backend
    - Backend used for pixel fitting. Currently can be 'scipyfit' or 'gpufit' (if pygpufit installed).
- fit_backend_comparison
    - Which backends to use in initial single pixel/ROI/AOI fit checks.
- force_fit
    - Overrides automatic reloading of previous fit results.
- fit_pl_pixels
    - False ignores the pixel fitting altogether, even if a previous fit result was found.
- scramble_pixels
    - Fit pixels in a random order to give a more accurate ETA (at no loss to fit speed).
- use_ROI_avg_fit_res_for_all_pixels
    - If 'false', uses init guesses below on each pixel, 'true' uses best fit to ROI average as initial guess.
- fit_functions
    - Dictionary. This one is important & required. The functions that make up your (pl) fit model. Each function type can be used multiple times. E.g. {"linear": 1, "lorentzian": 8} would give 8 lorentzian peaks with a linear (y= mx + c) background slope.
- param_guess
    - For some fit model parameter called 'param', this is the provided guess. Can be an array if you want different guesses for each of the 'n' functions in fit_functions (e.g. pos_0, pos_1, ... pos_7 for 8 lorentzians).
- param_range
    - Provide a range (+- guess) to bound fits for fit model parameter 'param'. 
- param_bounds
    - Manually provide bounds on fits for fit model parameter 'param'.

#### scipyfit options

- scipyfit_method
    - Method for scipy least squares fitting. Choose from: 'lm' (fast but doesn't use bounds), 
    'trf' (uses bounds and most reliable fit method but can be slow: default) and 'dogbox' (uses bounds and faster than trf).
- scipyfit_sub_threads
    - Number of threads to *not* use for fitting (e.g. throttle).
- scipyfit_show_progressbar
    - Display progressbar during fitting.
- scipyfit_use_analytic_jac
    - Use an analytically determined Jacobian, this should be faster but hasn't been implemented for every function. Default: True.
- scipyfit_verbose_fitting
    - Verbose fitting 0 = silent, 1 = term report, 2 display iterations.
- scipyfit_fit_jac_acc
    - If use_analytic_jac is false, what accuracy to numerically determine jacobian. The scheme ‘3-point’ is more accurate, but requires twice as many operations as '2-point' (default). The scheme 'cs' uses complex steps, and while potentially the most accurate, it is applicable only when the func correctly handles complex inputs and can be analytically continued to the complex plane.
- scipyfit_fit_gtol
    - Tolerance settings, the exact condition depends on method used - see scipy documentation. Defaults here: 1e-12. Tolerance for termination by the change of the independent variables. 
- scipyfit_fit_xtol
    - Tolerance for termination by the norm of the gradient.
- scipyfit_fit_ftol
    - Tolerance for termination by the change of the cost function.
- scipyfit_scale_x
    - Rescales the x by the units of the Jacobian (doesn't seem to make a difference)?
- scipyfit_loss_fn
    - Determines the loss function. This in non trivial check the scipy documentation. Default: linear. Options: linear, soft_l1, huber, cauchy, arctan.

#### gpufit options

- gpufit_tolerance
    - Setting a lower value for the tolerance results in more precise values for the fit parameters, but requires more fit iterations to reach convergence. A typical value for the tolerance settings is between 1.0E-3 and 1.0E-6. We use 1e-12 in scipy, so I'm using it here as the default too.
- gpufit_max_iterations
    - Max number of iterations. The maximum number of fit iterations permitted. If the fit has not converged after this number of iterations, the fit returns with a status value indicating that the maximum number of iterations was reached. Default: 25.
- gpufit_estimator_id
    - See https://gpufit.readthedocs.io/en/latest/fit_estimator_functions.html. Default: least squares estimator. Use MLE (maximum likelihood esitmator) for data subject to Poisson statistics -> noise in the data is assumed to be _purely_ Poissonian. (Options: LSE, MLE).

### Field retrieval parameters

- calc_field_pixels
    - Co you want to calculate field result for each pixel? (will auto reload prev result).
- force_field_calc
    - Force field calculate (don't reload prev. calculation).
- field_method
    -  Method to get bfield from pixel ODMR data
    - options:
        - "auto_dc": auto select from number of peaks (+ freqs_to_use) -> uses ham for 8 peaks, invert_unvs for 6 peaks, prop_single_bnv for 1,2
        - "hamiltonian_fitting": fit full hamiltonian
        - "prop_single_bnv": use fourier method to propagate bnv -> bxyz
        - "invert_unvs": take 3 unvs and (effective) approx_bxyz hamiltonian, but use simple inversion of unv matrix rather than fitting to hamiltonian
- hamiltonian
    - Type of hamiltonian to fit, if that bfield method is used. Or always fit if not 'bxyz' or 'approx_bxyz' assuming non-B parameters. Current options are in fact only 'bxyz' and 'approx_bxyz'.
- freqs_to_use
    - Which frequencies (e.g. lorentzian posns) to use for field retrieval. Must be len-8 iterable of values that evaluate as True/False (>=1 must be True).
- single_unv_choice
    - If option 'prop_single_bnv' given for 'field_method' but number of frequencies fit is 2 or 3, this option resolves ambiguity in which bnv to utilize. This option is used like so: single_bnv = bnvs[single_unv_choice].Note that `freqs_to_use` must still be set (to use 2 freqs only).
- diamond_ori
    - Diamond crystal orientation -> see qdmpy.constants. Default: HPHT orientation. Format: `<top face orientation>_<edge face orientation>`.
- auto_read_bias
    - Read magnetic field from options (i.e. if applied with vector electromagnet). --> system dependent option (i.e. Unimelb reads from metadata).
- auto_read_B
    - Guess mag field (Bx, By, Bz) from bias field parameters (bias_mag etc.).
- use_unvs
    - To specify unvs directly (below with "unvs" option) instead of guessing from bias_mag etc.
- unvs
    - Unit vectors of NV orientations, must be shape = (4,3). Equiv. to each NV frame's z axis. E.g. might look like `[[0.57735, 0.57735, 0.57735],
                           [-0.57735, 0.57735, 0.57735],
                           [0.57735, -0.57735, 0.57735],
                           [-0.57735, -0.57735, 0.57735]]`
- bias_mag
    - Field of bias field, used to automatically determine uNVs.
- bias_theta
    - In degrees.
- bias_phi
    - In degrees.
- field_param_guess
    - Same guess, range and bounds as above for pl fitting, but here for hamiltonian fitting. 
- field_param_range
- field_param_bounds


### Fourier parameters

- fourier_pad_mode
    - Padding mode, see np.pad. Defaults to mean
- fourier_pad_factor
    - Factor of image dimensions to pad by (on each side, e.g. 2 pad -> 2 either side of image)
- fourier_k_vector_epsilon
    - Add an epsilon value to the k-vectors to avoid some issues with 1/0. False/null/0 to not use.
- fourier_do_hanning_filter
    - Use hanning filter in fourier space (for all transformations).
- fourier_high_cutoff
    - Cutoff freqs (for bandpass) -> give as a distance/wavelength (k = 2pi/cutoff). Should be smaller number than low_cutoff
- fourier_low_cutoff
    - Cutoff freqs (for bandpass) -> give as a distance/wavelength (k = 2pi/cutoff). Should be larger number than high_cutoff

### Source recon parameters

- source_type
    - Type of source field to reconstruct, e.g. "current" or "magnetisation".
- standoff
    - Average distance NV layer <-> sample, or null/None to not use.
- nv_layer_thickness
    - Thickness of the NV layer, or null to not use. Disregarded if standoff is null.
- recon_methods
    - Reconstruction methods for B -> J or B -> M,  see e.g. Broadway 2020 http://dx.doi.org/10.1103/PhysRevApplied.14.024076. Must be an array of method you would like to run (even if only 1 choice). Choices: from_bxy, from_bz, from_bnv
- zero_point_normalisation_region
    - Region to use for zero-point normalisation. I.e. mean of this region is subtracted from Jx and Jy (or M etc.). (or null/none to not normalise). 
    - Format: [[x_top_left, y_top_left], [x_bot_right, y_bot_right], where y is from top to bottom.
- magnetisation_angle
    - magnetisation direction (intrinsic). Either zero (for z-magnetisation) or an angle in degress, from (+) x-axis towards (+) y-axis. untested.

### Background subtraction parameters

- bfield_bground_method
    - Method for bfield background subtraction (or null to avoid)
    - Descriptions
        - fix_zero: constant offset, set zero to chosen value (key 'zero' in bfield_bground_params)
        - three_point: plane background, calculated from three points (key 'points')
        - mean: constant offset of mean of image
        - poly: key 'order' polynomial fit to image
        - gaussian: gaussian fit to image
        - interpolate: image gaussian filtered (key 'sigma'), with data interpolated over
           polygons (key 'interp_method')
        - gaussian_filter: image gaussian filtered (key 'sigma')
- bfield_bground_params
    - Parameters for chosen bground method, required params are:
        - fix_zero: 'zero', a number
        - three_point: 'points' a length three iterable of [x, y] positions in image
        - mean: none
        - poly: key 'order', a natural number
        - gaussian: none
        - interpolate: key 'interp_method' (interpolation method in scipy griddata: nearest, linear,  cubic) and key 'sigma'
        - gaussian_filter: this one also requires sigma.
- bnv_bground_method
    - As above but for bnvs. This background subtraction is not carried on to analysis (just plotting & saving).
- bnv_bground_params
    - As above but for bnvs.
- source_bground_method
    - and/or use zero_point_normalisation_region (see above)
- source_bground_params

### Plotting parameters

- save_plots
- show_scalebar
- annotate_image_regions
    - Used for ROI/AOI pl plots only.
- save_fig_type
    - E.g. "png", "pdf", "svg" (svg slow!).
- large_fig_save_type
    - Jstream is huge & slow as an svg, allows large figs like this to have different type.
- colormaps
    - Choose colormap used for each type of image, a dict with possible keys:
        - param_images, residual_images, sigma_images, pl_images, bnv_images, dshift_images, bfield_images.
- colormap_range_dicts
    - Choose range of values mapped across to the colormap limits, for each 'type' of image if you want to override, you must copy the full dict (not just the ones you want to change).
    ```
        - Available options:
            type:
                    min_max                         : map between min and max of image
                    deviation_from_mean             : maps between (1 - dev) * mean and (1 + dev) * mean.
        deflt       min_max_symmetric_about_mean    : map symmetrically about mean, capturing all values
                    min_max_symmetric_about_zero    : map symmetrically about zero, capturing all values
                    percentile                      : maps the range between percentiles of the data
                    percentile_symmetric_about_zero : as above but symmetric about zero, caps. all vals
                    strict_range                    : maps colors between the values given

        values : float or array
            Different for each type:
                    min_max                         : not used
                    deviation_from_mean             : ('dev' above) float between 0 and 1
            deflt   min_max_symmetric_about_mean    : not used
                    min_max_symmetric_about_zero    : not used
                    percentile                      : list len 2 of (pref.) ints or floats in [0, 100]
                    percentile_symmetric_about_zero : list len 2 of (pref.) ints or floats in [0, 100]
                    strict_range                    : list len 2 of ints/floats

            - auto_sym_zero : bool
                - Optional boolean. If True, automatically detects if image has negative and positive values and makes colormap symmetric about zero. Only has an effect if the given c_range_type is "min_max" or "percentile". Defaults to True (i.e. if not specified)

        - All implemented in `qdmpy.plot.common._get_colormap_range`
    ```
- mpl_rcparams
    - Extra parameters (as dict) sent to matplotlib to define plotting stylesheet 
- polygon_patch_params
    - Parameters for annotating polygon patches onto images (passed to matplotlib.patches.Polygon)
- AOI_colors
    - Colors to identify with each AOI region (list of strings).
- fit_backend_colors
    - Colors to identify with each fit backend (dict of dicts).
- streamplot_options
    - fit options for streamplot
- streamplot_pl_alpha
    - alpha (i.e. transparency between 0 (invisible) and 1 (fully opaque)) of pl image behind behing streamplot


### Set internally

- cleaned
    - Was the options dict (after being read from .json) checked/cleaned?
- system
    - Pointer to system object.
- used_ref
    - Flag that we used a reference
- metadata
    - Any metadata read in.
- threads
    - For multiprocessing: # threads to use.
- rebinned_image_shape
    - Size of image (size_x, size_y) after rebinning, before being cut down to ROI.
- reloaded_prev_fit
    - Did we automatically reload a previous fit result?
- found_prev_result
    - Did we find a previous fit result (that matches current options)?
- found_prev_result_reason
    - Reasoning for above (str).
- output_dir
    - Directory to save results (directly in this dir: images).
- data_dir
    - Directory within output to save 'data' e.g. .txt files (output_dir/data_dir).
- fit_param_defn
    - For the record, the parameters (and units) used in this fit model.
- total_bin
    - Total (i.e. net) binning used.
- original_bin
    - E.g. camera binning (or labview binning).
- ModelID
    - modelID sent to gpufit (only used if gpufit selected as backend).
- CUDA_version_runtime
    - CUDA information, stored for the record.
- CUDA_version_driver
    - CUDA information, stored for the record.
- fit_time_(s)
    - Time to fit (pl) image.
- ham_fit_time_(s)
    - As above but for any hamiltonian fitting.
- field_dir
    - Field result directories
- field_sig_dir
- field_ref_dir
- field_method_used
    - Method used for bfield retrieval (hamiltonian_fitting etc.).
- field_params
    - Name of field params used in field retrieval model.
- found_prev_field_calc
    - Found previous field calculation result that matches current options.
- found_prev_field_calc_reason
    - Reasoning for above
- polygon_nodes
    - List of list of nodes, a list for each polygon.
- polygons
    - List of polygon objects


