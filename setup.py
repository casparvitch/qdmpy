import setuptools, pathlib

required = [
    "setuptools-git",
    "numpy",
    "matplotlib",
    "scipy",
    "numba",
    "psutil",
    "matplotlib-scalebar",
    "tqdm",
    "simplejson",
    "pandas",
    "rebin",
    "pyfftw",
    "PySimpleGUI",
]

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

if __name__ == "__main__":
    setuptools.setup(
        name="qdmpy",  # Replace with your own username
        version="0.2.0",
        author="Sam Scholten",
        author_email="samcaspar@gmail.com",
        description="Quantum Diamond MicroscoPy",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://gitlab.unimelb.edu.au/sscholten/qdmpy",
        keywords=["NV", "QDM", "Diamond", "Quantum", "Quantum Sensing", "gpufit"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation :: CPython",
            "License :: OSI Approved :: MIT License",
            "Development Status :: 2 - Pre-Alpha",
        ],
        license="MIT",
        package_dir={"": "src"},
        packages=setuptools.find_packages(
            where="src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
        ),
        install_requires=required,
        python_requires="==3.8.*",  # ">=3.8, <4", pyfftw currently <3.8 only...
        package_data={"": ["*.md", "*.json"]},
    )
# https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
