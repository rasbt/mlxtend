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
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import matplotlib; print('matplotlib %s' % matplotlib.__version__)"

if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s -v --with-coverage --cover-package=mlxtend.classifier
    nosetests -s -v --with-coverage --cover-package=mlxtend.evaluate
    nosetests -s -v --with-coverage --cover-package=mlxtend.math
    nosetests -s -v --with-coverage --cover-package=mlxtend.preprocessing
    nosetests -s -v --with-coverage --cover-package=mlxtend.regression
    nosetests -s -v --with-coverage --cover-package=mlxtend.text
    nosetests -s -v --with-coverage --cover-package=mlxtend.feature_selection
else
    nosetests -s -v mlxtend.classifier
    nosetests -s -v mlxtend.evaluate
    nosetests -s -v mlxtend.math
    nosetests -s -v mlxtend.preprocessing
    nosetests -s -v mlxtend.regression
    nosetests -s -v mlxtend.text
	nosetests -s -v mlxtend.feature_selection
fi
#make test-doc test-sphinxext
