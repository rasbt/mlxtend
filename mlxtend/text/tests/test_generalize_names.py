import sys

if sys.version_info < (3, 0):
    from nose.plugins.skip import SkipTest
    raise SkipTest

from mlxtend.text import generalize_names


def test_generalize_names():

    assert(generalize_names("Samuel Eto'o") == 'etoo s')
    assert(generalize_names("Eto'o, Samuel") == 'etoo s')
    assert(generalize_names("Eto'o, Samuel") == 'etoo s')
    assert(generalize_names('Xavi') == 'xavi')
    assert(generalize_names('Yaya Toure') == 'toure y')
    assert(generalize_names('Pozo, Jose Angel') == 'pozo j')
    assert(generalize_names('Pozo, Jose Angel') == 'pozo j')
    assert(generalize_names('Jose Angel Pozo') == 'pozo j')
    assert(generalize_names('Jose Pozo') == 'pozo j')
    assert(generalize_names('Pozo, Jose Angel', firstname_output_letters=2) ==
           'pozo jo')
    assert(generalize_names("Eto'o, Samuel", firstname_output_letters=2) ==
           'etoo sa')
    assert(generalize_names("Eto'o, Samuel", firstname_output_letters=0) ==
           'etoo')
    assert(generalize_names("Eto'o, Samuel", output_sep=', ') == 'etoo, s')
    assert(generalize_names("Eto'o, Samuel", output_sep=', ') == 'etoo, s')

    assert(generalize_names("van Persie, Robin", output_sep=', ') ==
           'vanpersie, r')
    assert(generalize_names("Robin van Persie", output_sep=', ') ==
           'vanpersie, r')
    assert(generalize_names("Rafael van der Vaart", output_sep=', ') ==
           'vandervaart, r')
    assert(generalize_names("van der Vaart, Rafael", output_sep=', ') ==
           'vandervaart, r')
    assert(generalize_names("Ben Hamer") == 'hamer b')
