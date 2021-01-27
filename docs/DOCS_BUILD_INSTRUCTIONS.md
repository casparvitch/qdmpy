To build documentation:

- navigate to the root directory, e.g. ~/src/QDMPy_proj/QDMPy_git/ (which should contain the directory QDMPy
- install pdoc: `pip3 install pdoc3` or similar (see [pdoc3](https://pdoc3.github.io/pdoc/))
- cmd: `pdoc --html --skip-errors --force --output-dir docs QDMPy`
- don't forget that period at the end there!

- the force option overwrites any existing docs
- the skip errors option raises warnings for import errors etc. (missing packages, e.g. if you haven't installed gpufit)
