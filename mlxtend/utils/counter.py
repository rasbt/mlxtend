# Sebastian Raschka 2014-2018
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
        Sets the number of decimal places when displaying the time elapsed in
        seconds.
    name : string (default: None)
        Prepends the specified name before the counter to allow distinguishing
        between multiple counters.

    Attributes
    ----------
    curr_iter : int
        The current iteration.
    start_time : float
        The system's time in seconds when the Counter was initialized.
    end_time : float
        The system's time in seconds when the Counter was last updated.

    Examples
    --------
    >>> cnt = Counter()
    >>> for i in range(20):
    ...     # do some computation
    ...     time.sleep(0.1)
    ...     cnt.update()
    20 iter | 2 sec
    >>> print('The counter was initialized.'
              ' %d seconds ago.' % (time.time() - cnt.start_time))
    The counter was initialized 2 seconds ago
    >>> print('The counter was last updated'
              ' %d seconds ago.' % (time.time() - cnt.end_time))
    The counter was last updated 0 seconds ago.

    For more usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/utils/Counter/

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

        if self.name is None:
            self._print_name = ''
        else:
            self._print_name = '%s: ' % self.name

        self.start_time = time.time()
        self.curr_iter = 0
        if start_newline:
            self.stream.write('\n')

    def update(self):
        """Print current iteration and time elapsed."""
        self.curr_iter += 1

        self.end_time = time.time()

        out = '%d iter | %s sec' % (self.curr_iter,
                                    self.precision % (self.end_time
                                                      - self.start_time))

        self.stream.write('\r%s%s' % (self._print_name, out))
        self.stream.flush()
