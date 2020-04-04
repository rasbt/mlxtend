#!/usr/bin/env bash

set -e

python --version
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import mlxtend; print('mlxtend %s' % mlxtend.__version__)"

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then 


        if [[ "$COVERAGE" == "true" ]]; then

            if [[ "$IMAGE" == "true" ]]; then
                 PYTHONPATH='.' pytest -sv --cov=mlxtend
            else
                 PYTHONPATH='.' pytest -sv --cov=mlxtend --ignore=mlxtend/image
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

    if [[ "$IMAGE" == "true" ]]; then
      python make_api.py 
      find sources -name "*.ipynb" -exec jupyter nbconvert --to notebook --execute {} \;
    else      
      python make_api.py --ignore_packages "mlxtend.image"
      find sources -name "*.ipynb" -not -path "sources/user_guide/image/*" -exec jupyter nbconvert --to notebook --execute {} \;
    
    fi
fi