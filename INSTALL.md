# usage

- follow instructions below for install
- copy one of the example notebooks/scipts to another directory
- open the directory in jupyterlab from a shell: `jupyter-lab`
- edit the options, see '/src/systems/unimelb_defaults.json' for docs

# examples/tests

- grab testing datasets from <TODO>

# basic install
(assuming you know git)

- [clone](https://gitlab.unimelb.edu.au/sscholten/qdmpy)
- install python3.8
- navigate to root-dir (containing 'src' etc.)
```bash
pip install . # will install deps
pip install jupyterlab
```

# if above install didn't work
- install conda/miniconda

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
conda install -c dataonlygreater python-polylabel 
pip install rebin foronoi # hopefully these don't muck anything up, I will be trying to remove these soon
```

- although the above may have issues, see e.g. [here](https://github.com/conda/conda-build/issues/4251)

# Git

- See README for now
- TODO add cleaner git instructions etc.

# gpufit

- [gitlab repo](https://gitlab.unimelb.edu.au/QSL/gpufit)
- instructions TODO, or see Defective Club Article

# docs

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


