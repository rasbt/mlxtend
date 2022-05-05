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

from .prog_class import Prog


class ProgPercent(Prog):
    """
    Initializes a progress bar object that allows visuzalization
    of an iterational computation in the standard output screen.

    Parameters
    ----------
    iterations : `int`
        Number of iterations for the iterative computation.
    track_time : `bool` (default: `True`)
        Prints elapsed time when loop has finished.
    stream : `int` (default: 2).
        Setting the output stream.
        Takes `1` for stdout, `2` for stderr, or a custom stream object
    title : `str` (default: `''`).
        Setting a title for the percentage indicator.
    monitor : `bool` (default: `False`)
        Monitors CPU and memory usage if `True` (requires `psutil` package).
    update_interval : float or int (default: `None`)
        The update_interval in seconds controls how often the progress
        is flushed to the screen.
        Automatic mode if `update_interval=None`.

    """

    def __init__(
        self,
        iterations,
        track_time=True,
        stream=2,
        title="",
        monitor=False,
        update_interval=None,
    ):
        Prog.__init__(
            self, iterations, track_time, stream, title, monitor, update_interval
        )
        self.last_progress = 0
        self._print()
        if monitor:
            try:
                self.process.cpu_percent()
                self.process.memory_percent()
            except AttributeError:  # old version of psutil
                self.process.get_cpu_percent()
                self.process.get_memory_percent()

    def _cache_percent_indicator(self, last_progress):
        self._cached_output += "[%3d %%]" % (last_progress)

    def _print(self, force_flush=False):
        """Prints formatted percentage and tracked time to the screen."""
        self._stream_flush()
        next_perc = self._calc_percent()
        if self.update_interval:
            do_update = time.time() - self.last_time >= self.update_interval
        elif force_flush:
            do_update = True
        else:
            do_update = next_perc > self.last_progress

        if do_update and self.active:
            self.last_progress = next_perc
            self._cache_percent_indicator(self.last_progress)
            if self.track:
                self._cached_output += " Time elapsed: " + self._get_time(
                    self._elapsed()
                )
                self._cache_eta()
            if self.item_id:
                self._cache_item_id()
            self._stream_out("\r%s" % self._cached_output)
            self._stream_flush()
            self._cached_output = ""
