"""Use in conjunction with line profiler to see what lines are being
the bottleneck in Cython code for cmf_newton_solver."""
import pycmf
from pycmf.cmf_solvers import _newton_update_left, _newton_update_V
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.special import expit

import line_profiler

import argparse
parser = argparse.ArgumentParser(description='Line profiling for Cython')
parser.add_argument('--sample', '-s', type=float, default=1.,
                    help='The stochastic gradient ratio')
args = parser.parse_args()


def assert_stats(profile, name):
    profile.print_stats()


rng = np.random.mtrand.RandomState(42)
X = np.abs(rng.randn(2000, 150))
X[:1000, 2 * np.arange(10) + 100] = 0
X[1000:, 2 * np.arange(10)] = 0
X = csr_matrix(X)
Y = expit(rng.randn(150, 10))

U, V = pycmf.cmf._initialize_mf(X, 10, random_state=42, non_negative=True)
V, Z = pycmf.cmf._initialize_mf(Y, 10, random_state=42, non_negative=True)

X = X.tocsr() if issparse(X) else np.ascontiguousarray(X)
Y = Y.T.tocsr() if issparse(Y) else np.ascontiguousarray(Y.T)

# order memory to be F-contiguous when passed to cython functions
U = np.ascontiguousarray(U)
V = np.ascontiguousarray(V)
Z = np.ascontiguousarray(Z)

# profile _newton_update_V
print("Profiling _newton_update_V")
profile = line_profiler.LineProfiler(_newton_update_V)
profile.runcall(_newton_update_V, U=U, V=V, Z=Z, X=X, Y=Y,
                alpha=0.1, l1_reg=0.5, l2_reg=0.5, non_negative=True,
                x_link="linear", y_link="logit", sg_sample_ratio=args.sample, hessian_pertubation=2.)
assert_stats(profile, _newton_update_V)

# profile _residual
print("Profiling _residual")
profile = line_profiler.LineProfiler(pycmf.cmf_newton_solver._residual)
profile.runcall(pycmf.cmf_newton_solver._residual,
                V[0, :], U.T, X[:, 0].T.tocsr(), "logit", True)
assert_stats(profile, pycmf.cmf_newton_solver._residual)

# profile _newton_update_left
print("Profiling _newton_update_U")
profile = line_profiler.LineProfiler(_newton_update_left)
profile.runcall(_newton_update_left, U=U, V=V, X=X,
                alpha=0.1, l1_reg=0., l2_reg=0.,
                link="linear", non_negative=True,
                sg_sample_ratio=args.sample, hessian_pertubation=2.)
assert_stats(profile, _newton_update_left)

# profle _stochastic_sample
print("Profiling _stochastic_sample")
profile = line_profiler.LineProfiler(pycmf.cmf_newton_solver._stochastic_sample)
profile.runcall(pycmf.cmf_newton_solver._stochastic_sample,
                U, X, args.sample, 0)
assert_stats(profile, pycmf.cmf_newton_solver._stochastic_sample)
