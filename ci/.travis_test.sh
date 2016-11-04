#!/usr/bin/env bash

set -e

if [[ "$TENSORFLOW" == "true" ]]; then
    if [[ "$COVERAGE" == "true" ]]; then
        nosetests -s -v mlxtend/tf_classifier --nologcapture --with-coverage
        nosetests -s -v mlxtend/tf_regressor --nologcapture --with-coverage
        nosetests -s -v mlxtend/tf_cluster --nologcapture --with-coverage
    else
        nosetests -s -v mlxtend/tf_classifier --nologcapture
        nosetests -s -v mlxtend/tf_regressor --nologcapture
        nosetests -s -v mlxtend/tf_cluster --nologcapture
    fi
else
    if [[ "$COVERAGE" == "true" ]]; then
        nosetests -s -v --with-coverage --exclude-dir=mlxtend/tf_classifier --exclude-dir=mlxtend/tf_regressor --exclude-dir=mlxtend/tf_cluster --exclude-dir=mlxtend/plotting
    else
        nosetests -s -v --exclude-dir=mlxtend/tf_classifier --exclude-dir=mlxtend/tf_regressor --exclude-dir=mlxtend/tf_cluster --exclude-dir=mlxtend/plotting
    fi
fi

if [[ "$NOTEBOOKS" == "true" ]]; then
    cd docs
    python make_api.py
    jupyter nbconvert --to notebook --execute docs/sources/user_guide/classifier/Perceptron.ipynb
fi
