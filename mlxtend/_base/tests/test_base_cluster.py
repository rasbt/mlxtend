# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend._base import _BaseCluster
import numpy as np
from mlxtend.utils import assert_raises


def test_init():
    cl = _BaseCluster(print_progress=0, random_seed=1)
