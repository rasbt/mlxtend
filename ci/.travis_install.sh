#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD



set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
#export CC=gcc
#export CXX=g++

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
wget http://repo.continuum.io/miniconda/Miniconda-3.9.1-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda

# Configure the conda environment and put it in the path using the
# provided versions


if [[ "$LATEST" == "true" ]]; then
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy scipy scikit-learn cython pandas matplotlib
else
    conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
        scikit-learn=$SKLEARN_VERSION \
	      pandas=$PANDAS_VERSION \
	     	matplotlib=$MATPLOTLIB_VERSION cython
fi


source activate testenv

pip install nose-exclude

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

if [[ "$TENSORFLOW" == "true" ]]; then
  # workaround for "Cannot remove entries from nonexistent file /home/travis/miniconda/envs/testenv/lib/python3.4/site-packages/easy-install.pth"
  conda update setuptools --yes
  curl https://bootstrap.pypa.io/ez_setup.py -o - | python
  case "$TRAVIS_OS_NAME" in
      "linux")
          case "$PYTHON_VERSION" in
              "2.7")
                  pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp27-none-linux_x86_64.whl
              ;;
              "3.4")
                  pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl
              ;;
          esac
      ;;
      "osx")
          case "$PYTHON_VERSION" in
              "2.7")
                  pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp27-none-any.whl
              ;;
              "3.5")
                  pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.7.1-cp35-none-any.whl
              ;;
          esac
      ;;
  esac
fi

# Build mlxtend in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.

if [[ "$TENSORFLOW" == "true" ]]; then
  python -c "import tensorflow; print('tensorflow %s' % tensorflow.__version__)"
else
  python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
  python -c "import pandas; print('pandas %s' % pandas.__version__)"
  python -c "import matplotlib; print('matplotlib %s' % matplotlib.__version__)"
fi

# install mlxtend
python setup.py install #build_ext --inplace
