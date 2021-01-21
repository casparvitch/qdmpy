Quantum Diamond MicroscoPy
==========================



# Environment management

Pipenv (add link)

install instructions at (link). Need python installed, use pipx method they quote.


(cd to directory that contains .git directory)
`cd QDMPy` 

`pipenv install`


to start working (in virtual environment):
`pipenv shell`


# Installation

NOPE Sam, SDK 8.1 for VS2015: https://developer.microsoft.com/en-us/windows/downloads/sdk-archive/

### Saving widget state

In jupyterlab > settings > advanced settings editor > jupyter widgets > saveState: true

### Exporting notebook to pdf
- install [pandoc](https://pandoc.org/installing.html)
- need to change to %matplotlib inline?
	- check Github issues (e.g. [#16](https://github.com/matplotlib/ipympl/issues/16), [#150](https://github.com/matplotlib/ipympl/issues/150) and [#176](https://github.com/matplotlib/ipympl/pull/176))


## GPUFit

Source: [https://github.com/gpufit/Gpufit.git](https://github.com/gpufit/Gpufit.git)


### Windows

Required:
CUDA-capable graphics card, updated Nvidia graphics driver.

#### Cuda installation

TODO

#### GPUFit installation

Can install a binary from [here](https://github.com/gpufit/Gpufit/releases).

However this does not include the model functions we require (e.g. lorentzians, exponential decays) so you will need to write these into the source and rebuild it (each time you want a new model function...). Details below.

- [ ] Goal: keep our own branch with our model functions, rebase on master from time to time.
	- This way we can keep a group-wide collection of usable model functions
	- store in gitlab?
	- [multiple remotes, github](https://forum.sublimetext.com/t/working-with-multiple-remotes/53489/3)
- [ ] Keep in separate repo to QDMPy

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
 - `QDMPy.<this module name (filename)>.ClassNameOne`

Functions
---------
 - `QDMPy.<this module name (filename)>.Function1`
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
        Normalised measurement array, shape: [sweep_list, x, y].

    sweep_list : np array, 1D
        Affine parameter list (e.g. tau or freq)

    fit_model : `QDMPy.fit_models.FitModel` object.

    Returns
    -------
    `QDMPy.fit_shared.ROIAvgFitResult` object containing the fit result (see class specifics)
    """
```

To build documentation (html) using [pdoc](https://pdoc3.github.io/pdoc/doc/pdoc/#pdoc) tool:
- navigate to dir: QDMPy_git/QDMPy, e.g. for Sam: ~/src/nv/QDMPy_proj/QDMPy_git/QDMPy
- cmd: `pdoc --html --force --output-dir doc .`
- the period at the end of the command above is required!