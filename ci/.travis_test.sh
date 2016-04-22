#!/usr/bin/env bash

set -e

if [ "$TENSORFLOW" == "true" ]; then
    if [[ "$COVERAGE" == "true" ]]; then
        nosetests -s -v mlxtend.tf_classifier --nologcapture --with-coverage
    else
        nosetests -s -v mlxtend.tf_classifier --nologcapture
else
    if [[ "$COVERAGE" == "true" ]]; then
        nosetests -s -v --with-coverage --exclude-dir=mlxtend/tf_classifier --exclude-dir=mlxtend/data --exclude-dir=mlxtend/general_plotting
    else
        nosetests -s -v --exclude-dir=mlxtend/tf_classifier --exclude-dir=mlxtend/data --exclude-dir=mlxtend/general_plotting
    fi
fi
