# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Base Iterative Model (Iterative Model Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from sys import stderr
from time import time

import numpy as np


class _IterativeModel(object):
    def __init__(self):
        pass

    def _shuffle_arrays(self, arrays):
        """Shuffle arrays in unison."""
        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]

    def _print_progress(self, iteration, n_iter, cost=None, time_interval=10):
        if self.print_progress > 0:
            s = "\rIteration: %d/%d" % (iteration, n_iter)
            if cost:
                s += " | Cost %.2f" % cost
            if self.print_progress > 1:
                if not hasattr(self, "ela_str_"):
                    self.ela_str_ = "00:00:00"
                if not iteration % time_interval:
                    ela_sec = time() - self._init_time
                    self.ela_str_ = self._to_hhmmss(ela_sec)
                s += " | Elapsed: %s" % self.ela_str_
                if self.print_progress > 2:
                    if not hasattr(self, "eta_str_"):
                        self.eta_str_ = "00:00:00"
                    if not iteration % time_interval:
                        eta_sec = (ela_sec / float(iteration)) * n_iter - ela_sec
                        if eta_sec < 0.0:
                            eta_sec = 0.0
                        self.eta_str_ = self._to_hhmmss(eta_sec)
                    s += " | ETA: %s" % self.eta_str_
            stderr.write(s)
            stderr.flush()

    def _to_hhmmss(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def _yield_minibatches_idx(self, rgen, n_batches, data_ary, shuffle=True):
        indices = np.arange(data_ary.shape[0])

        if shuffle:
            indices = rgen.permutation(indices)
        if n_batches > 1:
            remainder = data_ary.shape[0] % n_batches

            if remainder:
                minis = np.array_split(indices[:-remainder], n_batches)
                minis[-1] = np.concatenate((minis[-1], indices[-remainder:]), axis=0)
            else:
                minis = np.array_split(indices, n_batches)

        else:
            minis = (indices,)

        for idx_batch in minis:
            yield idx_batch

    def _init_params(
        self,
        weights_shape,
        bias_shape=(1,),
        random_seed=None,
        dtype="float64",
        scale=0.01,
        bias_const=0.0,
    ):
        """Initialize weight coefficients."""
        rgen = np.random.RandomState(random_seed)
        w = rgen.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        if bias_const != 0.0:
            b += bias_const
        return b.astype(dtype), w.astype(dtype)
