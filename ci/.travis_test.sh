#!/usr/bin/env bash

set -e


if [[ "$COVERAGE" == "true" ]]; then

    if [[ "$IMAGE" == "true" ]]; then
        nosetests -s -v --with-coverage
    else
        nosetests -s -v --with-coverage --exclude-dir=mlxtend/image
    fi

else
    if [[ "$IMAGE" == "true" ]]; then
        nosetests -s -v
    else
        nosetests -s -v --exclude-dir=mlxtend/image
    fi
fi


if [[ "$NOTEBOOKS" == "true" ]]; then
    cd docs
    python make_api.py
    find sources -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute {} \;
fi
