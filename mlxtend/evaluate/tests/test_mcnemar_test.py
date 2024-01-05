# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_almost_equal

from mlxtend.evaluate import mcnemar
from mlxtend.utils import assert_raises


def test_input_dimensions():
    t = np.ones((3, 3))
    assert_raises(ValueError, "Input array must be a 2x2 array.", mcnemar, t)


def test_defaults():
    tb = np.array([[101, 121], [59, 33]])

    chi2, p = (20.672222222222221, 5.4500948254271171e-06)
    chi2p, pp = mcnemar(tb)

    assert_almost_equal(chi2, chi2p, decimal=7)
    assert_almost_equal(p, pp, decimal=7)


def test_corrected_false():
    tb = np.array([[101, 121], [59, 33]])
    chi2, p = (21.355555555555554, 3.8151358651125936e-06)
    chi2p, pp = mcnemar(tb, corrected=False)

    assert_almost_equal(chi2, chi2p, decimal=7)
    assert_almost_equal(p, pp, decimal=7)


def test_exact():
    tb = np.array([[101, 121], [59, 33]])

    p = 4.4344492637555101e-06
    chi2p, pp = mcnemar(tb, exact=True)

    assert chi2p is None
    assert_almost_equal(p, pp, decimal=7)
    assert p < 4.45e-06


def test_exact_corrected():
    tb = np.array([[101, 121], [59, 33]])

    p = 4.4344492637555101e-06
    chi2p, pp = mcnemar(tb, exact=True, corrected=False)

    assert chi2p is None
    assert_almost_equal(p, pp, decimal=7)
    assert p < 4.45e-06
