#!/usr/bin/env bash

set -e


if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s -v --with-coverage
fi

if [[ "$NOTEBOOKS" == "true" ]]; then
    cd docs
    python make_api.py
    find sources -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute {} \;
fi
