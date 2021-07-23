To build documentation:

- navigate to the root directory, e.g. ~/src/qdmpy_proj/qdmpy_git/ (which should contain the directory 'src')
- install pdoc: `pip3 install pdoc3` or similar (see [pdoc3](https://pdoc3.github.io/pdoc/))
- cmd: `pdoc3 --output-dir docs --html --template-dir ./docs/ --force ./src/qdmpy`
- (or similar, may be different on windows)

- the force option overwrites any existing docs
- you may want to use skip errors option raises warnings for import errors etc. (missing packages, e.g. if you haven't installed gpufit)
