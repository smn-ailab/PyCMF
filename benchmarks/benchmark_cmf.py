import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import expit
import timeit
import sys; sys.path.append("../..")
from CMF import CMF

import argparse
parser = argparse.ArgumentParser(description='Benchmark CMF')
parser.add_argument('--runs', '-r', type=int, default=1,
                    help='The number of runs per experiment')
args = parser.parse_args()

class BenchmarkCase:
    def __init__(self, test_name, function_name, arguments=""):
        self.test_name = test_name
        self.function_name = function_name
        self.arguments = arguments
        
    def run(self):
        print("-" * 75)
        t = timeit.Timer(f"{self.function_name}({self.arguments})",
                         setup=f"from __main__ import {self.function_name}")
        best_run = min(t.repeat(args.runs))
        print(f"{self.test_name}: Best of {args.runs} runs was {best_run} sec.")

def dense_cmf_benchmark(solver):
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(2000, 150))
    Y = np.abs(rng.randn(150, 10))
    model = CMF(n_components=10, solver=solver,
                random_state=42)
    U, V, Z = model.fit_transform(X, Y)

def dense_cmf_with_logits_benchmark():
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(2000, 150))
    Y = np.abs(rng.randn(150, 10))
    model = CMF(n_components=10, solver="newton",
                random_state=42)
    U, V, Z = model.fit_transform(X, Y)

def sparse_cmf_benchmark(solver):
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(2000, 150))
    X[:1000, 2 * np.arange(10) + 100] = 0
    X[1000:, 2 * np.arange(10)] = 0
    X_sparse = csc_matrix(X)
    Y = np.abs(rng.randn(150, 10))
    model = CMF(n_components=10, solver=solver,
                random_state=42)
    U, V, Z = model.fit_transform(X, Y)

def sparse_cmf_with_logits_benchmark():
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(2000, 150))
    X[:1000, 2 * np.arange(10) + 100] = 0
    X[1000:, 2 * np.arange(10)] = 0
    X_sparse = csc_matrix(X)
    Y = expit(rng.randn(150, 10))
    model = CMF(n_components=10, solver="newton",
                random_state=42)
    U, V, Z = model.fit_transform(X, Y)

if __name__ == "__main__":
    print("=" * 75)
    print("Commencing benchmark...")
    for solver in ["mu", "newton"]:
        BenchmarkCase(f"Dense CMF, {solver}", "dense_cmf_benchmark", arguments=f"'{solver}'").run()
    BenchmarkCase(f"Dense CMF with logits", "dense_cmf_with_logits_benchmark").run()
    for solver in ["mu", "newton"]:
        BenchmarkCase(f"Sparse CMF, {solver}", "sparse_cmf_benchmark", arguments=f"'{solver}'").run()
    BenchmarkCase(f"Sparse CMF with logits", "sparse_cmf_with_logits_benchmark").run()
    print("=" * 75)