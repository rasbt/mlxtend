import sys

if sys.version_info < (3, 0):
    from nose.plugins.skip import SkipTest
    raise SkipTest

from data_names import csv
from mlxtend.text import generalize_names_duplcheck
from mlxtend.text import generalize_names
from io import StringIO
import pandas as pd


def test_generalize_names_duplcheck():

    df = pd.read_csv(StringIO(csv))

    # duplicates before
    dupl = any(df['Name'].apply(generalize_names).duplicated())
    assert dupl is True

    # no duplicates
    df_new = generalize_names_duplcheck(df=df, col_name='Name')
    no_dupl = any(df_new['Name'].duplicated())
    assert no_dupl is False
