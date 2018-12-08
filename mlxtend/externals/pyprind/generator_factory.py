# PyPrind
# Author: Sebastian Raschka <mail@sebastianraschka.com>
# Contributors: https://github.com/rasbt/pyprind/graphs/contributors
# License: BSD 3 clause
# Code Repository: https://github.com/rasbt/pyprind
# PyPI: https://pypi.python.org/pypi/PyPrind

from .progbar import ProgBar
from .progpercent import ProgPercent


def generator_factory(mother_class):
    def generator_progress(iteritem, iterations=None, *args, **kw):
        if iterations is None:
            iterations = len(iteritem)
        assert iterations
        mbar = mother_class(iterations, *args, **kw)
        for item in iteritem:
            yield item
            mbar.update()
    return generator_progress

prog_percent = generator_factory(ProgPercent)
prog_bar = generator_factory(ProgBar)
