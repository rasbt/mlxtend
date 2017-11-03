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
    name : string (default: None)
        Prepends the specified name before the counter to allow distinguishing
        between multiple counters.

    Attributes
    ----------
    curr_iter : int
        The current iteration.
    start_time : int
        The system's time in seconds when the Counter was initialized.

    """
    def __init__(self, stderr=False, start_newline=True, name=None):
        if stderr:
            self.stream = sys.stderr
        else:
            self.stream = sys.stdout
        self.name = name
        self.start_time = time.time()
        self.curr_iter = 0
        if start_newline:
            self.stream.write('\n')

    def update(self):
        """Print current iteration and time elapsed."""
        self.curr_iter += 1
        out = '%d iter | %.2f sec' % (self.curr_iter, time.time() -
                                      self.start_time)
        if self.name is None:
            self.stream.write('\r%s' % out)
        else:
            self.stream.write('\r %s: %s' % (self.name, out))
        self.stream.flush()
