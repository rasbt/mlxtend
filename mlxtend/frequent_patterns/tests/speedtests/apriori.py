# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

"""Speedtest for apriori candidate generation.

Run from the repository root:

    uv run --group dev python mlxtend/frequent_patterns/tests/speedtests/apriori.py

To compare the working tree implementation against HEAD:

    uv run --group dev python mlxtend/frequent_patterns/tests/speedtests/apriori.py --compare-head
"""

import argparse
import importlib
import subprocess
import time
import types

import numpy as np
import pandas as pd

PRESETS = {
    "quick": {
        "rows": 2400,
        "groups": 18,
        "items_per_group": 5,
        "groups_per_transaction": 5,
        "min_support": 0.045,
        "max_len": 4,
        "repeats": 7,
    },
    "realistic": {
        "rows": 10000,
        "groups": 40,
        "items_per_group": 6,
        "groups_per_transaction": 5,
        "min_support": 0.03,
        "max_len": 4,
        "repeats": 5,
    },
}


def make_dataset(
    seed=123,
    n_rows=10000,
    n_groups=40,
    items_per_group=6,
    groups_per_transaction=5,
):
    rng = np.random.default_rng(seed)
    n_cols = n_groups * items_per_group
    ary = np.zeros((n_rows, n_cols), dtype=bool)

    for row in range(n_rows):
        groups = rng.choice(n_groups, size=groups_per_transaction, replace=False)
        for group in groups:
            start = group * items_per_group
            size = rng.integers(2, items_per_group + 1)
            items = rng.choice(items_per_group, size=size, replace=False)
            ary[row, start + items] = True

    columns = ["i%d" % idx for idx in range(n_cols)]
    return pd.DataFrame(ary, columns=columns)


def load_head_apriori_module():
    source = subprocess.check_output(
        ["git", "show", "HEAD:mlxtend/frequent_patterns/apriori.py"], text=True
    )
    module = types.ModuleType("mlxtend.frequent_patterns.apriori_head")
    module.__package__ = "mlxtend.frequent_patterns"
    exec(compile(source, "apriori_head.py", "exec"), module.__dict__)
    return module


def normalize_result(result):
    return sorted(
        (tuple(sorted(itemset)), round(float(support), 12))
        for support, itemset in zip(result["support"], result["itemsets"])
    )


def run_benchmark(label, apriori_func, df, kwargs, repeats):
    times = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = apriori_func(df, **kwargs)
        times.append(time.perf_counter() - start)

    print("%s rows: %d" % (label, result.shape[0]))
    print("%s times: %s" % (label, " ".join("%.4f" % t for t in times)))
    print("%s median: %.4f" % (label, np.median(times)))
    return result, times


def run_candidate_benchmark(label, generate_new_combinations, old_combinations):
    start = time.perf_counter()
    generated = np.fromiter(generate_new_combinations(old_combinations), dtype=int)
    elapsed = time.perf_counter() - start
    count = generated.size // (old_combinations.shape[1] + 1)

    print("%s candidate count: %d" % (label, count))
    print("%s candidate time: %.6f" % (label, elapsed))


def run(args):
    current = importlib.import_module("mlxtend.frequent_patterns.apriori")
    modules = [("current", current)]
    if args.compare_head:
        modules.insert(0, ("head", load_head_apriori_module()))

    print("preset: %s" % args.preset)
    print(
        "dataset: rows=%d, columns=%d, groups=%d, items_per_group=%d, "
        "groups_per_transaction=%d"
        % (
            args.rows,
            args.groups * args.items_per_group,
            args.groups,
            args.items_per_group,
            args.groups_per_transaction,
        )
    )
    print(
        "apriori: min_support=%.4f, max_len=%d, repeats=%d"
        % (args.min_support, args.max_len, args.repeats)
    )

    df = make_dataset(
        seed=args.seed,
        n_rows=args.rows,
        n_groups=args.groups,
        items_per_group=args.items_per_group,
        groups_per_transaction=args.groups_per_transaction,
    )
    base_kwargs = {
        "min_support": args.min_support,
        "use_colnames": False,
        "max_len": args.max_len,
    }

    for low_memory in [False, True]:
        print("\nlow_memory: %s" % low_memory)
        kwargs = dict(base_kwargs, low_memory=low_memory)
        baseline_result = None
        baseline_times = None

        for label, module in modules:
            result, times = run_benchmark(
                label, module.apriori, df, kwargs, repeats=args.repeats
            )
            if baseline_result is None:
                baseline_result = result
                baseline_times = times
            else:
                same_output = normalize_result(baseline_result) == normalize_result(
                    result
                )
                speedup = np.median(baseline_times) / np.median(times)
                print("same output vs %s: %s" % (modules[0][0], same_output))
                print("speedup vs %s: %.2fx" % (modules[0][0], speedup))

        if low_memory:
            continue

        old_combinations = np.array(
            sorted(
                tuple(sorted(itemset))
                for itemset in baseline_result["itemsets"]
                if len(itemset) == 3
            )
        )
        if old_combinations.size == 0:
            continue

        print("\ncandidate generation from frequent 3-itemsets")
        for label, module in modules:
            run_candidate_benchmark(
                label, module.generate_new_combinations, old_combinations
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="realistic")
    parser.add_argument("--compare-head", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--items-per-group", type=int)
    parser.add_argument("--groups-per-transaction", type=int)
    parser.add_argument("--min-support", type=float)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args()

    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


if __name__ == "__main__":
    run(parse_args())
