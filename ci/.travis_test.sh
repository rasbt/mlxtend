#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"


if [[ "$TENSORFLOW" == "true" ]]; then
    python -c "import tensorflow; print('tensorflow %s' % tensorflow.__version__)"
    nosetests -s -v mlxtend.tf_classifier --nologcapture
else
    python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
    python -c "import pandas; print('pandas %s' % pandas.__version__)"
    python -c "import matplotlib; print('matplotlib %s' % matplotlib.__version__)"

    if [[ "$COVERAGE" == "true" ]]; then
        nosetests -s -v --with-coverage --exclude-dir=mlxtend/tf_classifier --exclude-dir=mlxtend/data --exclude-dir=mlxtend/general_plotting

    else
        nosetests -s -v --exclude-dir=mlxtend/tf_classifier --exclude-dir=mlxtend/data --exclude-dir=mlxtend/general_plotting
    fi

fi
