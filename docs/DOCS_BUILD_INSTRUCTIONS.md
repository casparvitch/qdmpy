To build documentation:

- navigate to the root directory, e.g. ~/src/qdmpy_proj/qdmpy_git/ (which should contain the directory qdmpy)
- install pdoc: `pip3 install pdoc3` or similar (see [pdoc3](https://pdoc3.github.io/pdoc/))
- cmd: `pdoc --output-dir docs --html --config latex_math=True --force qdmpy`
- don't forget that period at the end there!

- the force option overwrites any existing docs
- the skip errors option raises warnings for import errors etc. (missing packages, e.g. if you haven't installed gpufit)
