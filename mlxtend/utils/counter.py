# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# A counter class for printing the progress of an iterator.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import time
import sys


class Counter(object):

    """Class to display the progress of for-loop iterators.
    Parameters
    ----------
    stderr : bool (default: True)
        Prints output to sys.stderr if True; uses sys.stdout otherwise.
    start_newline : bool (default: True)
        Prepends a new line to the counter, which prevents overwriting counters
        if multiple counters are printed in succession.
    precision: int (default: 0)
        Sets the precison of the displayed iteration time.
    name : string (default: None)
        Prepends the specified name before the counter to allow distinguishing
        between multiple counters.
    Attributes
    ----------
    curr_iter : int
        The current iteration.
    start_time : int
        The system's time in seconds when the Counter was initialized.
    iteration_time : int
        The system's time in seconds when the Counter was last updated.
    """
    def __init__(self, stderr=False, start_newline=True, precision=0,
                 name=None):
        if stderr:
            self.stream = sys.stderr
        else:
            self.stream = sys.stdout
        if isinstance(precision, int):
            self.precision = '%%.%df' % precision
        else:
            self.precision = '%d'
        self.name = name
        self.start_time = time.time()
        self.iteration_time = time.time()
        self.curr_iter = 0
        if start_newline:
            self.stream.write('\n')

    def update(self):
        """Print current iteration and time elapsed."""
        self.curr_iter += 1
        self.iteration_time = time.time()
        out = '%d iter | %s sec' % (self.curr_iter,
                                    self.precision % self.total_elapsed())
        if self.name is None:
            self.stream.write('\r%s' % out)
        else:
            self.stream.write('\r %s: %s' % (self.name, out))
        self.stream.flush()

    def total_elapsed(self):
        """Return time elapsed since initialized."""
        return time.time() - self.start_time

    def iteration_elapsed(self):
        """Return time elapsed since last updated."""
        return time.time() - self.iteration_time
