# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .remove_chartjunk import remove_borders
from .scatter import category_scatter
from .stacked_barplot import stacked_barplot
from .enrichment_plot import enrichment_plot

__all__ = ["remove_borders", "category_scatter",
           "stacked_barplot", "enrichment_plot"]
