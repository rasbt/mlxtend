#!/usr/bin/env bash

set -e

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then 
    if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then

        wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    fi
else
    if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then

        wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    fi
fi
    

bash miniconda.sh -b -p "$HOME/miniconda"
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda update -q pip
conda info -a

conda create -q -n test-environment python="$MINICONDA_PYTHON_VERSION"
source activate test-environment

if [ "${LATEST}" = "true" ]; then
    conda install numpy scipy pandas scikit-learn joblib
else
    conda install numpy="$NUMPY_VERSION" scipy="$SCIPY_VERSION" pandas="$PANDAS_VERSION" joblib="$JOBLIB_VERSION"
    # temporary fix because 0.22 cannot be installed from the main conda branch
    conda install scikit-learn="$SKLEARN_VERSION" -c conda-forge
fi


conda install matplotlib
conda install seaborn
conda install jupyter
conda install pytest


if [ "${IMAGE}" = "true" ]; then

    if [ "${LATEST}" = "true" ]; then
        pip install dlib
        pip install imageio
        pip install scikit-image
    else
        pip install dlib=="$DLIB_VERSION"
        pip install imageio=="$IMAGEIO_VERSION"
        pip install scikit-image=="$SKIMAGE_VERSION"
    fi
fi

if [ "${COVERAGE}" = "true" ]; then
    conda install pytest-cov
    conda install coveralls
fi

python --version
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import mlxtend; print('mlxtend %s' % mlxtend.__version__)"
