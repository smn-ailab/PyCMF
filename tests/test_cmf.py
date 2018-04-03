"""Using sklearn internal testing tools for the sake of later adding as a feature"""
import numpy as np
import scipy.sparse as sp
from scipy import linalg
import itertools
import pytest

from pycmf import CMF, collective_matrix_factorization, analysis
import sklearn.decomposition.nmf as nmf
from scipy.sparse import csr_matrix
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_raise_message, assert_no_warnings
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.extmath import squared_norm
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer

solvers = ["mu", "newton"]


def test_input_shape_compatibility_check():
    X = np.ones((5, 2))
    Y = np.ones((5, 2))
    msg = "Expected X.shape[1] == Y.shape[0], " + \
          "found X.shape = {}, Y.shape = {}".format(X.shape, Y.shape)
    assert_raise_message(ValueError, msg, CMF(solver='mu',
                                              beta_loss=2).fit, X, Y)


# ignore UserWarning raised when both solver='mu' and init='nndsvd'
@ignore_warnings(category=UserWarning)
@pytest.mark.parametrize("solver", solvers)
def test_fit_nn_output(solver):
    # Test that the decomposition does not contain negative values
    X = np.c_[5 * np.ones(5) - np.arange(1, 6),
              5 * np.ones(5) + np.arange(1, 6)]
    Y = np.c_[5 * np.ones(5) - np.arange(1, 6),
              5 * np.ones(5) + np.arange(1, 6)].T
    for init in (None, 'nndsvd', 'nndsvda', 'nndsvdar', 'random'):
        model = CMF(n_components=2, solver=solver, x_init=init, y_init=init,
                    random_state=0)
        U, V, Z = model.fit_transform(X, Y)
        assert_false((U < 0).any() or
                     (V < 0).any() or
                     (Z < 0).any())


@pytest.mark.parametrize("solver", solvers)
def test_fit_close(solver):
    rng = np.random.mtrand.RandomState(42)
    # Test that the fit is not too far away
    for rndm_state in [0]:
        pnmf = CMF(n_components=5, solver=solver, x_init='nndsvdar', y_init='nndsvdar',
                   random_state=rndm_state, max_iter=1000)
        X = np.abs(rng.randn(6, 5))
        Y = np.abs(rng.randn(5, 6))
        assert_less(pnmf.fit(X, Y).reconstruction_err_, 0.1)


def test_transform_custom_init():
    # Smoke test that checks if CMF.fit_transform works with custom initialization
    random_state = np.random.RandomState(0)
    X = np.abs(random_state.randn(6, 5))
    Y = np.abs(random_state.randn(5, 1))
    n_components = 4
    avg = np.sqrt(X.mean() / n_components)
    U_init = np.abs(avg * random_state.randn(6, n_components))
    V_init = np.abs(avg * random_state.randn(5, n_components))
    avg = np.sqrt(Y.mean() / n_components)
    Z_init = np.abs(avg * random_state.randn(1, n_components))

    m = CMF(solver='newton', n_components=n_components,
            x_init='custom', y_init='custom',
            random_state=0)
    m.fit_transform(X, Y, U=U_init, V=V_init, Z=Z_init)


def test_input_method_compatibility():
    # Smoke test for combinations between different init methods
    rng = np.random.mtrand.RandomState(0)
    X = np.abs(rng.randn(6, 5))
    Y = np.abs(rng.randn(5, 6))
    n_components = 4
    avg = np.sqrt(X.mean() / n_components)
    U_init = np.abs(avg * rng.randn(6, n_components))
    V_init = np.abs(avg * rng.randn(5, n_components))
    avg = np.sqrt(Y.mean() / n_components)
    Z_init = np.abs(avg * rng.randn(6, n_components))
    inits = [None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom']
    for x_init, y_init in itertools.product(inits, inits):
        pnmf = CMF(n_components=n_components, solver='mu',
                   x_init=x_init, y_init=y_init,
                   random_state=0, max_iter=1)
        pnmf.fit_transform(X, Y, U=U_init, V=V_init, Z=Z_init)


def test_n_components_greater_n_features():
    # Smoke test for the case of more components than features.
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(30, 10))
    Y = np.abs(rng.randn(10, 15))
    CMF(n_components=15, random_state=0, tol=1e-2).fit(X, Y)


@pytest.mark.parametrize("solver", solvers)
def test_recover_low_rank_matrix(solver):
    rng = np.random.mtrand.RandomState(42)
    # Test that the fit is not too far away
    pnmf = CMF(5, solver=solver, x_init='nndsvdar', y_init='nndsvdar',
               random_state=0, max_iter=1000)
    U = np.abs(rng.randn(10, 5))
    V = np.abs(rng.randn(8, 5))
    Z = np.abs(rng.randn(6, 5))
    X = np.dot(U, V.T)
    Y = np.dot(V, Z.T)
    assert_less(pnmf.fit(X, Y).reconstruction_err_, 1.0)


@ignore_warnings(category=ConvergenceWarning)
def test_loss_decreasing():
    # test that the objective function for at least one of the matrices is decreasing
    n_components = 10
    alpha = 0.1
    tol = 0.

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(20, 15))
    Y = np.abs(rng.randn(15, 10))
    U0, V0 = nmf._initialize_nmf(X, n_components, init='random',
                                 random_state=42)
    V0_, Z0 = nmf._initialize_nmf(Y, n_components, init='random',
                                  random_state=42)
    V0 = (V0.T + V0_) / 2

    U, V, Z = U0.copy(), V0.copy(), Z0.copy()

    # since Hessian is being perturbed, might not have to work for newton-raphson solver
    for solver in ['mu']:

        previous_x_loss = nmf._beta_divergence(X, U, V.T, 2)
        previous_y_loss = nmf._beta_divergence(Y, V, Z.T, 2)
        for _ in range(30):
            # one more iteration starting from the previous results
            U, V, Z, _ = collective_matrix_factorization(
                X, Y, U, V, Z, x_init='custom', y_init='custom',
                n_components=n_components, max_iter=1,
                solver=solver, tol=tol, verbose=0, random_state=0)

            x_loss = nmf._beta_divergence(X, U, V.T, 2)
            y_loss = nmf._beta_divergence(Y, V, Z.T, 2)
            max_loss_decrease = max(previous_x_loss - x_loss, previous_y_loss - y_loss)
            assert_greater(max_loss_decrease, 0)
            previous_x_loss = x_loss
            previous_y_loss = y_loss


@pytest.mark.parametrize("solver", solvers)
def test_l1_regularization(solver):
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(6, 5))
    Y = np.abs(rng.randn(5, 4))

    # L1 regularization should increase the number of zeros
    l1_reg = 2.
    reg = CMF(n_components=n_components, solver=solver,
              l1_reg=l1_reg, random_state=42)
    model = CMF(n_components=n_components, solver=solver,
                l1_reg=0., random_state=42)

    U_reg, V_reg, Z_reg = reg.fit_transform(X, Y)
    U_model, V_model, Z_model = model.fit_transform(X, Y)

    U_reg_n_zeros = U_reg[U_reg == 0].size
    V_reg_n_zeros = V_reg[V_reg == 0].size
    Z_reg_n_zeros = Z_reg[Z_reg == 0].size
    U_model_n_zeros = U_model[U_model == 0].size
    V_model_n_zeros = V_model[V_model == 0].size
    Z_model_n_zeros = Z_model[Z_model == 0].size

    msg = "solver: {}".format(solver)

    # If one matrix is full of zeros,
    # it might make sense for the other matrices to reduce the number of zeros
    # Therefore, we compare the total number of zeros
    assert_greater(U_reg_n_zeros + V_reg_n_zeros + Z_reg_n_zeros,
                   U_model_n_zeros + V_model_n_zeros + Z_model_n_zeros, msg)


@pytest.mark.parametrize("solver", solvers)
def test_l2_regularization(solver):
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(6, 5))
    Y = np.abs(rng.randn(5, 4))
    # L2 regularization should decrease the mean of the coefficients
    l2_reg = 2.
    model = CMF(n_components=n_components, solver=solver,
                l2_reg=0., random_state=42)
    reg = CMF(n_components=n_components, solver=solver,
              l2_reg=l2_reg, random_state=42)

    U_reg, V_reg, Z_reg = reg.fit_transform(X, Y)
    U_model, V_model, Z_model = model.fit_transform(X, Y)

    msg = "solver: {}".format(solver)
    assert_greater(U_model.mean(), U_reg.mean(), msg)
    assert_greater(V_model.mean(), V_reg.mean(), msg)
    assert_greater(Z_model.mean(), Z_reg.mean(), msg)


def test_nonnegative_condition_for_newton_solver():
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(6, 5))
    Y = np.abs(rng.randn(5, 4))

    model = CMF(n_components=n_components, solver="newton",
                l2_reg=0., random_state=42,
                U_non_negative=False, V_non_negative=False, Z_non_negative=False)

    U, V, Z = model.fit_transform(X, Y)

    # if one value is negative in any matrix, since X and Y are non-negative,
    # all the other matrices will need to have negative values
    assert_less(np.min(U), 0)
    assert_less(np.min(V), 0)
    assert_less(np.min(Z), 0)


def test_logit_link_optimization():
    n_components = 5
    rng = np.random.mtrand.RandomState(42)
    X = 1 / (1 + np.exp(-rng.randn(6, 5)))
    Y = 1 / (1 + np.exp(-rng.randn(5, 4)))

    model = CMF(n_components=n_components, solver="newton",
                l2_reg=0., random_state=42, x_link="logit", y_link="logit",
                U_non_negative=False, V_non_negative=False, Z_non_negative=False)

    U, V, Z = model.fit_transform(X, Y)
    assert_less(model.reconstruction_err_, 0.1)


def test_logit_link_non_negative_optimization():
    # Test if the logit link function works with a non-negative counterpart
    n_components = 5
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(6, 5)
    X[X < 0] = 0
    Y = 1 / (1 + np.exp(-rng.randn(5, 4)))

    model = CMF(n_components=n_components, solver="newton",
                l2_reg=0., random_state=42, y_link="logit",
                U_non_negative=True, V_non_negative=True, Z_non_negative=False,
                hessian_pertubation=0.2, max_iter=1000)

    U, V, Z = model.fit_transform(X, Y)
    assert_less(model.reconstruction_err_, 0.1)


@pytest.mark.parametrize("solver", solvers)
def test_sparse_input(solver):
    # Test that sparse matrices are accepted as input
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0
    A_sparse = csr_matrix(A)
    B = np.abs(rng.randn(10, 5))
    B[2 * np.arange(5), :] = 0
    B_sparse = csr_matrix(B)

    est1 = CMF(solver=solver, n_components=5,
               x_init='random', y_init='random',
               random_state=0, tol=1e-2)
    est2 = clone(est1)

    U1, V1, Z1 = est1.fit_transform(A, B)
    U2, V2, Z2 = est2.fit_transform(A_sparse, B_sparse)

    assert_array_almost_equal(U1, U2)
    assert_array_almost_equal(V1, V2)
    assert_array_almost_equal(Z1, Z2)


def test_stochastic_newton_solver():
    rng = np.random.mtrand.RandomState(42)

    model = CMF(n_components=5, solver="newton", x_init='svd', y_init='svd',
                U_non_negative=False, V_non_negative=False, Z_non_negative=False, alpha=0.5,
                sg_sample_ratio=0.5, random_state=0, max_iter=1000)
    X = rng.randn(6, 5)
    Y = rng.randn(5, 6)
    assert_less(model.fit(X, Y).reconstruction_err_, 0.1)


def test_stochastic_newton_solver_sparse_input_close():
    rng = np.random.mtrand.RandomState(42)

    model = CMF(n_components=5, solver="newton", x_init='svd', y_init='svd',
                U_non_negative=False, V_non_negative=False, Z_non_negative=False, alpha=0.5,
                sg_sample_ratio=0.5, random_state=0, max_iter=1000)
    A = rng.randn(6, 5)
    B = rng.randn(5, 6)
    A_sparse = csr_matrix(A)
    B_sparse = csr_matrix(B)

    assert_less(model.fit(A_sparse, B_sparse).reconstruction_err_, 0.1)


def test_stochastic_newton_solver_sparse_input():
    rng = np.random.mtrand.RandomState(36)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0
    A_sparse = csr_matrix(A)
    B = np.abs(rng.randn(10, 5))
    B[2 * np.arange(5), :] = 0
    B_sparse = csr_matrix(B)

    est1 = CMF(n_components=5, solver="newton", x_init='svd', y_init='svd',
               U_non_negative=False, V_non_negative=False, Z_non_negative=False,
               sg_sample_ratio=0.5, random_state=0, max_iter=1000)
    est2 = clone(est1)

    U1, V1, Z1 = est1.fit_transform(A, B)
    U2, V2, Z2 = est2.fit_transform(A_sparse, B_sparse)

    assert_array_almost_equal(U1, U2)
    assert_array_almost_equal(V1, V2)
    assert_array_almost_equal(Z1, Z2)


@ignore_warnings(category=UserWarning)
def test_svd_ncomponents_lt_nfeatures():
    # smoke test for input where a dimension is 1
    rng = np.random.mtrand.RandomState(42)

    model = CMF(n_components=3, solver="newton", x_init='svd', y_init='svd',
                U_non_negative=False, V_non_negative=False, Z_non_negative=False,
                random_state=0, max_iter=1)
    X = rng.randn(6, 4)
    Y = rng.randn(4, 2)
    model.fit(X, Y)


def test_auto_compute_alpha():
    rng = np.random.mtrand.RandomState(36)
    X = rng.randn(10, 10)
    Y = rng.randn(10, 5)

    x_emphasis_model = CMF(n_components=2, solver="newton", x_init='svd', y_init='svd',
                           U_non_negative=False, V_non_negative=False, Z_non_negative=False,
                           random_state=0, max_iter=100, alpha=0.5)
    # automatic = weight * number_of_elements is constant for both X and Y
    y_emphasis_model = CMF(n_components=2, solver="newton", x_init='svd', y_init='svd',
                           U_non_negative=False, V_non_negative=False, Z_non_negative=False,
                           random_state=0, max_iter=100, alpha="auto")

    U1, V1, Z1 = x_emphasis_model.fit_transform(X, Y)
    U2, V2, Z2 = y_emphasis_model.fit_transform(X, Y)

    assert_greater(np.linalg.norm(np.dot(U2, V2.T) - X), np.linalg.norm(np.dot(U1, V1.T) - X))
    assert_greater(np.linalg.norm(np.dot(V1, Z1.T) - Y), np.linalg.norm(np.dot(V2, Z2.T) - Y))


@pytest.mark.xfail
@pytest.mark.parametrize("solver", solvers)
def test_transform_after_fit(solver):
    rng = np.random.mtrand.RandomState(36)
    X = rng.randn(7, 5)
    Y = rng.randn(5, 3)
    X_new = rng.randn(7, 10)

    fit_model = CMF(n_components=2, solver=solver, x_init='svd', y_init='svd',
                    U_non_negative=False, V_non_negative=False, Z_non_negative=False,
                    random_state=0, max_iter=100)
    fit_transform_model = clone(fit_model)

    fit_model.fit(X, Y)
    U_f, V_f, Z_f = fit_model.transform(X, None)
    U_ft, V_ft, Z_ft = fit_transform_model.fit_tranform(X, Y)


def test_analysis():
    # smoke test to see that analysis works
    rng = np.random.mtrand.RandomState(36)
    model = CMF(n_components=2, solver="newton", max_iter=1)
    c = CountVectorizer()
    X_ = c.fit_transform(["hello world",
                          "goodbye world",
                          "hello goodbye"])
    X_ = csr_matrix(X_)
    Y = np.abs(rng.randn(3, 1))
    model.fit_transform(X_.T, Y)
    model.print_topic_terms(c, importances=False)
    model.print_topic_terms(c, importances=True)
