#!/usr/bin/env bash

set -e


if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s -v --with-coverage --exclude-dir=mlxtend/tf_classifier --exclude-dir=mlxtend/tf_regressor --exclude-dir=mlxtend/tf_cluster --exclude-dir=mlxtend/plotting
else
    nosetests -s -v --exclude-dir=mlxtend/tf_classifier --exclude-dir=mlxtend/tf_regressor --exclude-dir=mlxtend/tf_cluster --exclude-dir=mlxtend/plotting
fi

if [[ "$NOTEBOOKS" == "true" ]]; then
    cd docs
    python make_api.py
    find sources -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute {} \;
fi
