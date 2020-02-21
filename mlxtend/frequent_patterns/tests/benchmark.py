# Sebastian Raschka 2014-2019
# myxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pandas as pd
import numpy as np
import gzip
import os
import sys
from time import time
import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def raise_timeout(signum, frame):
    raise TimeoutError


files = [
    # "chess.dat.gz",
    # "connect.dat.gz",
    "mushroom.dat.gz",
    "pumsb.dat.gz",
    "pumsb_star.dat.gz",
    # "T10I4D100K.dat.gz",
    # "T40I10D100K.dat.gz",
    # "kosarak.dat.gz", # this file is too large in sparse format
    # "kosarak-1k.dat.gz",
    # "kosarak-10k.dat.gz",
    # "kosarak-50k.dat.gz",
    # "kosarak-100k.dat.gz",
    # "kosarak-200k.dat.gz",
]


low_memory = True
commit = "b731fd2"
test_supports = [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001]

for sparse, col_major in [[False, True], [False, False], [True, True]]:
    sys.stdout = open("Results/{}-sparse{}-col_major{}.out".format(
                          commit, sparse, col_major), "w")
    for filename in files:
        with gzip.open(os.path.join("data", filename)) if filename.endswith(
            ".gz"
        ) else open(os.path.join("data", filename)) as f:
            data = f.readlines()

        dataset = [list(map(int, line.split())) for line in data]
        items = np.unique([item for itemset in dataset for item in itemset])
        print("{} contains {} transactions and {} items".format(
                  filename, len(dataset), len(items)))

        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset, sparse=sparse)
        columns = ["c"+str(i) for i in te.columns_]
        if sparse:
            try:
                df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=columns)
            except AttributeError:
                # pandas < 0.25
                df = pd.SparseDataFrame(te_ary, columns=columns,
                                        default_fill_value=False)
        else:
            df = pd.DataFrame(te_ary, columns=columns)
            if col_major:
                df = pd.DataFrame({col: df[col] for col in df.columns})
        np.info(df.values)

        kwds = {"use_colnames": False, "low_memory": low_memory}
        for min_support in test_supports:
            tick = time()
            with timeout(120):
                print(apriori(df, min_support=min_support, verbose=1, **kwds))
            print("\nmin_support={} temps: {}\n".format(
                      min_support, time() - tick))
            if time() - tick < 10:
                times = []
                for _ in range(5):
                    tick = time()
                    apriori(df, min_support=min_support, verbose=0, **kwds)
                    times.append(time() - tick)
                print("Times:", times)
    sys.stdout.close()
