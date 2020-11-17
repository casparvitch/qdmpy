from setuptools import setup, find_packages

install_requires = [
    "tqdm",
    "numpy",
    "pathlib",
    "matplotlib",
    # "Pillow",
    # "PySimpleGUI",
    # "adjustText",
    # "snakeviz",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

    setup(
        name="QDMPy",
        version="0.0.1",
        description="Quantum Diamond MicroscoPy",
        long_description=long_description,
        long_description_content_type="text/markdown",
        # entry_points={"console_scripts": ["flti=flti.command_line:nmap_gui"]},
        url="https://github.com/casparvitch/QDMPy",
        author="casparvitch",
        author_email="samcaspar@gmail.com",
        license="MIT",
        packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
        install_requires=install_requires,
        zip_safe=False,
        include_package_data=True,
    )
