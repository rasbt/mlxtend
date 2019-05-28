#!/usr/bin/env bash

set -e


if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then 


        if [[ "$COVERAGE" == "true" ]]; then

            if [[ "$IMAGE" == "true" ]]; then
                 PYTHONPATH='.' pytest -sv --with-coverage
            else
                 PYTHONPATH='.' pytest -sv --with-coverage --ignore=mlxtend/image
            fi

        else
            if [[ "$IMAGE" == "true" ]]; then
                 PYTHONPATH='.' pytest -sv
            else
                 PYTHONPATH='.' pytest -sv --ignore=mlxtend/image
            fi
        fi

else
     PYTHONPATH='.' pytest -sv --ignore=mlxtend/plotting
fi
  

if [[ "$NOTEBOOKS" == "true" ]]; then
    cd docs
    python make_api.py
    find sources -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute {} \;
fi
