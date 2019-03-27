# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from .bootstrap import bootstrap
from .bootstrap_outofbag import BootstrapOutOfBag
from .bootstrap_point632 import bootstrap_point632_score
from .cochrans_q import cochrans_q
from .confusion_matrix import confusion_matrix
from .feature_importance import feature_importance_permutation
from .lift_score import lift_score
from .mcnemar import mcnemar_table
from .mcnemar import mcnemar_tables
from .mcnemar import mcnemar
from .permutation import permutation_test
from .scoring import scoring
from .ttest import paired_ttest_resampled
from .ttest import paired_ttest_kfold_cv
from .ttest import paired_ttest_5x2cv
from .holdout import RandomHoldoutSplit
from .holdout import PredefinedHoldoutSplit
from .f_test import ftest
from .f_test import combined_ftest_5x2cv
from .proportion_difference import proportion_difference
from .bias_variance_decomp import bias_variance_decomp

__all__ = ["scoring", "confusion_matrix",
           "mcnemar_table", "mcnemar_tables",
           "mcnemar", "lift_score",
           "bootstrap", "permutation_test",
           "BootstrapOutOfBag", "bootstrap_point632_score",
           "cochrans_q", "paired_ttest_resampled",
           "paired_ttest_kfold_cv", "paired_ttest_5x2cv",
           "feature_importance_permutation",
           "RandomHoldoutSplit", "PredefinedHoldoutSplit",
           "ftest", "combined_ftest_5x2cv",
           "proportion_difference", "bias_variance_decomp"]
