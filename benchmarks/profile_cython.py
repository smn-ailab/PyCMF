import pstats, cProfile
import sys; sys.path.append("..") # TODO: Make installation work and remove this sh*t
import pycmf
from pycmf.cmf_solvers import _newton_update_U, _newton_update_V, _newton_update_Z
import numpy as np
from scipy.sparse import csc_matrix

import line_profiler

import argparse
parser = argparse.ArgumentParser(description='Line profiling for Cython')
parser.add_argument('--sample', '-s', type=float, default=1.,
                    help='The stochastic gradient ratio')
args = parser.parse_args()

def assert_stats(profile, name):
    profile.print_stats()
    stats = profile.get_stats()
    # assert len(stats.timings) > 0, "No profile stats."
    # for key, timings in stats.timings.items():
    #     if key[-1] == name:
    #         assert len(timings) > 0
    #         break
    # else:
    #     raise ValueError("No stats for %s." % name)

rng = np.random.mtrand.RandomState(42)
X = csc_matrix(np.abs(rng.randn(1000, 150)))
Y = np.abs(rng.randn(150, 10))

U, V = pycmf.cmf._initialize_mf(X, 10, random_state=42, non_negative=True)
V, Z = pycmf.cmf._initialize_mf(Y, 10, random_state=42, non_negative=True)

from functools import partial
print("Profiling _newton_update_V")
profile = line_profiler.LineProfiler(_newton_update_V)
profile.runcall(_newton_update_V, U=U, V=V, Z=Z, X=X, Y=Y,
               alpha=0.1, l1_reg=0.5, l2_reg=0.5, non_negative=True,
               x_link="linear", y_link="logit", sg_sample_ratio=args.sample, hessian_pertubation=2.)
assert_stats(profile, _newton_update_V)

print("Profiling _residual")
profile = line_profiler.LineProfiler(pycmf.cmf_newton_solver._residual)
profile.runcall(pycmf.cmf_newton_solver._residual,
                V[0, :], U.T, X[:, 0].T, "linear", True)
assert_stats(profile, pycmf.cmf_newton_solver._residual)

print("Profiling _newton_update_U")
profile = line_profiler.LineProfiler(_newton_update_U)
profile.runcall(_newton_update_U, U=U, V=V, X=X,
                alpha=0.1, l1_reg=0., l2_reg=0.,
                link="linear", non_negative=True,
                sg_sample_ratio=1., hessian_pertubation=2.)
assert_stats(profile, _newton_update_U)

print("Profiling _newton_update_Z")
profile = line_profiler.LineProfiler(_newton_update_Z)
profile.runcall(_newton_update_Z, Z=Z, V=V, Y=Y,
                alpha=0.1, l1_reg=0., l2_reg=0., 
                link="logit", non_negative=True,
                sg_sample_ratio=args.sample, hessian_pertubation=2.)
assert_stats(profile, _newton_update_U)

print("Profiling _stochastic_sample")
profile = line_profiler.LineProfiler(pycmf.cmf_newton_solver._stochastic_sample)
profile.runcall(pycmf.cmf_newton_solver._stochastic_sample,
                U, X, args.sample, 0)
assert_stats(profile, pycmf.cmf_newton_solver._stochastic_sample)
