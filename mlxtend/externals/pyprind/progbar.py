"""
Sebastian Raschka 2014-2017
Python Progress Indicator Utility

Author: Sebastian Raschka <sebastianraschka.com>
License: BSD 3 clause

Contributors: https://github.com/rasbt/pyprind/graphs/contributors
Code Repository: https://github.com/rasbt/pyprind
PyPI: https://pypi.python.org/pypi/PyPrind
"""


import time
from math import floor

from .prog_class import Prog


class ProgBar(Prog):
    """
    Initializes a progress bar object that allows visuzalization
    of an iterational computation in the standard output screen.

    Parameters
    ----------
    iterations : `int`
        Number of iterations for the iterative computation.
    track_time : `bool` (default: `True`)
        Prints elapsed time when loop has finished.
    width : `int` (default: 30)
        Sets the progress bar width in characters.
    stream : `int` (default: 2).
        Setting the output stream.
        Takes `1` for stdout, `2` for stderr, or a custom stream object
    title : `str` (default:  `''`)
        Setting a title for the progress bar.
    monitor : `bool` (default: False)
        Monitors CPU and memory usage if `True` (requires `psutil` package).
    update_interval : float or int (default: None)
        The update_interval in seconds controls how often the progress
        is flushed to the screen.
        Automatic mode if update_interval=None.

    """

    def __init__(
        self,
        iterations,
        track_time=True,
        width=30,
        bar_char="#",
        stream=2,
        title="",
        monitor=False,
        update_interval=None,
    ):
        Prog.__init__(
            self, iterations, track_time, stream, title, monitor, update_interval
        )
        self.bar_width = width
        self._adjust_width()
        self.bar_char = bar_char
        self.last_progress = 0

        if monitor:
            try:
                self.process.cpu_percent()
                self.process.memory_percent()
            except AttributeError:  # old version of psutil
                self.process.get_cpu_percent()
                self.process.get_memory_percent()
        if self.item_id:
            self._cache_item_id()

    def _adjust_width(self):
        """Shrinks bar if number of iterations is less than the bar width"""
        if self.bar_width > self.max_iter:
            self.bar_width = int(self.max_iter)
            # some Python 3.3.3 users specifically
            # on Linux Red Hat 4.4.7-1, GCC v. 4.4.7
            # reported that self.max_iter was converted to
            # float. Thus this fix to prevent float multiplication of chars.

    def _cache_progress_bar(self, progress):
        remaining = self.bar_width - progress
        self._cached_output += "0% [{}{}] 100%".format(
            self.bar_char * int(progress), " " * int(remaining)
        )

    def _print(self, force_flush=False):
        progress = floor(self._calc_percent() / 100 * self.bar_width)
        if self.update_interval:
            do_update = time.time() - self.last_time >= self.update_interval
        elif force_flush:
            do_update = True
        else:
            do_update = progress > self.last_progress

        if do_update and self.active:
            self._cache_progress_bar(progress)
            if self.track:
                self._cache_eta()
            if self.item_id:
                self._cache_item_id()
            self._stream_out("\r%s" % self._cached_output)
            self._stream_flush()
            self._cached_output = ""
        self.last_progress = progress
