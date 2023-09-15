# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from os.path import abspath, dirname, join, realpath

from setuptools import find_packages, setup

PROJECT_ROOT = dirname(realpath(__file__))


def get_version(rel_path):
    here = abspath(dirname(__file__))
    with open(join(here, rel_path), "r") as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

setup(
    name="mlxtend",
    version=get_version("mlxtend/__init__.py"),
    description="Machine Learning Library Extensions",
    author="Sebastian Raschka",
    author_email="mail@sebastianraschka.com",
    url="https://github.com/rasbt/mlxtend",
    packages=find_packages(),
    package_data={
        "": ["LICENSE-BSD3.txt", "LICENSE-CC-BY.txt", "README.md", "requirements.txt"]
    },
    include_package_data=True,
    install_requires=install_reqs,
    extras_require={"testing": ["pytest"], "docs": ["mkdocs"]},
    license="BSD 3-Clause",
    platforms="any",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    long_description="""

A library of Python tools and extensions for data science.


Contact
=============

If you have any questions or comments about mlxtend,
please feel free to contact me via
eMail: mail@sebastianraschka.com
or Twitter: https://twitter.com/rasbt

This project is hosted at https://github.com/rasbt/mlxtend

The documentation can be found at https://rasbt.github.io/mlxtend/

""",
)
