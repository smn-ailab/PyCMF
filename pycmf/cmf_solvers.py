import numbers
import time

import numpy as np
import scipy

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition.nmf import _beta_divergence, _beta_loss_to_float
from scipy.special import expit
from scipy.sparse import issparse
from .cmf_newton_solver import _newton_update_U, _newton_update_V, _newton_update_Z

EPSILON = np.finfo(np.float32).eps

INTEGER_TYPES = (numbers.Integral, np.integer)

USE_CYTHON = True

# utility functions
def sigmoid(M):
    return expit(M)


def d_sigmoid(M):
    sgm = sigmoid(M)
    return sgm * (1 - sgm)


def inverse(x, link):    
    if link == "linear":
        return x
    elif link == "logit":
        return sigmoid(x)
    else:
        raise ValueError("Invalid link function {}".format(link))


def compute_factorization_error(target, left_factor, right_factor, link, beta_loss):
    if link == "linear":
        return _beta_divergence(target, left_factor, right_factor, beta_loss, square_root=True)
    elif link == "logit":
        return np.linalg.norm(target - sigmoid(np.dot(left_factor, right_factor)))


class _IterativeCMFSolver:
    """Boilerplate for iterative solvers (mu and newton)
        Implement the update_step method in concrete subclasses to use.
        Parameters
        ----------
        tol : float, default: 1e-4
            Tolerance of the stopping condition.

        max_iter : integer, default: 200
            Maximum number of iterations before timing out.

        l1_reg : double, default: 0.
            L1 regularization parameter. Currently same for all matrices.

        l2_reg : double, default: 0.
            L2 regularization parameter

        alpha: double, default: 0.5
            Determines trade-off between optimizing for X and Y.
            The larger the value, the more X is prioritized in optimization.

        beta_loss : float or string, default 'frobenius'
            Currently disabled. Used only in 'mu' solver.
            String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
            Beta divergence to be minimized, measuring the distance between X
            and the dot product WH. Note that values different from 'frobenius'
            (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
            fits.

        update_H : boolean, default: True
            Currently disabled. Need to enable in future to implement transform method in CNMF.

        verbose : integer, default: 0
            The verbosity level.

        U_non_negative: bool, default: True
            Whether to enforce non-negativity for U. Only applicable for the newton solver.

        V_non_negative: bool, default: True
            Whether to enforce non-negativity for V. Only applicable for the newton solver.

        Z_non_negative: bool, default: True
            Whether to enforce non-negativity for Z. Only applicable for the newton solver.

        x_link: str, default: "linear"
            One of either "logit" of "linear". The link function for transforming UV^T to approximate X

        y_link: str, default: "linear"
            One of either "logit" of "linear". The link function for transforming VZ^T to approximate Y

        hessian_pertubation: double, default: 0.2
            The pertubation to the Hessian in the newton solver to maintain positive definiteness
        """
    def __init__(self, max_iter=200, tol=1e-4, beta_loss="frobenius",
                 l1_reg=0, l2_reg=0, alpha=0.5,
                 update_H=True, verbose=0,
                 U_non_negative=True, V_non_negative=True, Z_non_negative=True,
                 x_link="linear", y_link="linear", hessian_pertubation=0.2, 
                 sg_sample_ratio=1., random_state=None):
        self.max_iter = max_iter
        self.tol = tol
        self.beta_loss = _beta_loss_to_float(beta_loss)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.update_H = update_H
        self.verbose = verbose
        self.U_non_negative = U_non_negative
        self.V_non_negative = V_non_negative
        self.Z_non_negative = Z_non_negative
        self.x_link = x_link
        self.y_link = y_link
        self.hessian_pertubation = hessian_pertubation
        self.sg_sample_ratio = sg_sample_ratio
        if random_state is not None:
            np.random.seed(random_state)

    def update_step(self, X, Y, U, V, Z, l1_reg, l2_reg, alpha):
        """A single update step for all the matrices in the factorization."""
        raise NotImplementedError("Implement in concrete subclass to use")

    def fit_iterative_update(self, X, Y, U, V, Z):
        """Compute CNMF with iterative methods.
        The objective function is minimized with an alternating minimization of U, V
        and Z. Regularly prints error and stops update when improvement stops.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            First data matrix to be decomposed

        Y : {array-like, sparse matrix}, shape (n_features, n_labels)
            Second data matrix to be decomposed

        U : array-like, shape (n_samples, n_components)

        V : array-like, shape (n_features, n_components)

        Z : array-like, shape (n_labels, n_components)

        Returns
        -------
        U : array, shape (n_samples, n_components)
            Transformed data.

        V : array, shape (n_features, n_components)
            Transformed data.

        Z : array, shape (n_labels, n_components)
            Transformed data.

        n_iter : int
            The number of iterations done by the algorithm.
        """
        start_time = time.time()
        # TODO: handle beta loss other than fnorm

        previous_error = error_at_init = self.alpha * compute_factorization_error(X, U, V.T, self.x_link, self.beta_loss) + \
                                        (1 - self.alpha) * compute_factorization_error(Y, V, Z.T, self.y_link, self.beta_loss)

        for n_iter in range(1, self.max_iter + 1):

            self.update_step(X, Y, U, V, Z, self.l1_reg, self.l2_reg, self.alpha)

            # test convergence criterion every 10 iterations
            if self.tol > 0 and n_iter % 10 == 0:
                error = self.alpha *  compute_factorization_error(X, U, V.T, self.x_link, self.beta_loss) + \
                        (1 - self.alpha ) * compute_factorization_error(Y, V, Z.T, self.y_link, self.beta_loss)

                if self.verbose:
                    iter_time = time.time()
                    print("Epoch %02d reached after %.3f seconds, error: %f" %
                          (n_iter, iter_time - start_time, error))

                improvement_stopped = (previous_error - error) / error_at_init < self.tol
                if improvement_stopped:
                    break

                previous_error = error

        # do not print if we have already printed in the convergence test
        if self.verbose and (self.tol == 0 or n_iter % 10 != 0):
            end_time = time.time()
            print("Epoch %02d reached after %.3f seconds." %
                  (n_iter, end_time - start_time))

        return U, V, Z, n_iter


class MUSolver(_IterativeCMFSolver):
    """Internal solver that solves by iteratively multiplying the matrices element wise.
    The multiplying factors are always positive, meaning this solver can only return positive matrices.

    References
    ----------
    Wang, Y., Yanchunzhangvueduau, E., & Zhou, B. (n.d.).
    Semi-supervised collective matrix factorization for topic detection and document clustering.

    Lee, D., & Seung, H. (2001). Algorithms for non-negative matrix factorization.
    Advances in Neural Information Processing Systems, (1), 556–562.
    https://doi.org/10.1109/IJCNN.2008.4634046
    """

    @classmethod
    def _regularized_delta(cls, numerator, denominator, l1_reg, l2_reg, gamma, H):
        # Add L1 and L2 regularization
        if l1_reg > 0:
            denominator += l1_reg
        if l2_reg > 0:
            denominator = denominator + l2_reg * H
        denominator[denominator == 0] = EPSILON

        numerator /= denominator
        delta = numerator

        # gamma is in ]0, 1]
        if gamma != 1:
            delta **= gamma

        return delta

    @classmethod
    def _multiplicative_update_u(cls, X, U, V, beta_loss, l1_reg, l2_reg, gamma):
        numerator = safe_sparse_dot(X, V)
        denominator = np.dot(np.dot(U, V.T), V)
        return cls._regularized_delta(numerator, denominator, l1_reg, l2_reg, gamma, U)

    @classmethod
    def _multiplicative_update_z(cls, Y, V, Z, beta_loss, l1_reg, l2_reg, gamma):
        numerator = safe_sparse_dot(Y.T, V)
        denominator = np.dot(np.dot(Z, V.T), V)
        return cls._regularized_delta(numerator, denominator, l1_reg, l2_reg, gamma, Z)

    @classmethod
    def _multiplicative_update_v(cls, X, Y, U, V, Z, beta_loss, l1_reg, l2_reg, gamma):
        numerator = safe_sparse_dot(X.T, U) + safe_sparse_dot(Y, Z)
        denominator = np.dot(V, (np.dot(U.T, U) + np.dot(Z.T, Z)))
        return cls._regularized_delta(numerator, denominator, l1_reg, l2_reg, gamma, V)

    def update_step(self, X, Y, U, V, Z, l1_reg, l2_reg, alpha):
        # TODO: Enable specification of gamma
        gamma = 1.

        delta_V = self._multiplicative_update_v(X, Y, U, V, Z, self.beta_loss, l1_reg,
                                                l2_reg, gamma)
        V *= delta_V

        delta_U = self._multiplicative_update_u(X, U, V, self.beta_loss, l1_reg, l2_reg, gamma)
        U *= delta_U

        delta_Z = self._multiplicative_update_z(Y, V, Z, self.beta_loss, l1_reg, l2_reg, gamma)
        Z *= delta_Z

        
class NewtonSolver(_IterativeCMFSolver):
    """Internal solver that solves using the Newton-Raphson method.
    """
    def _newton_update_U(self, U, V, X, alpha, l1_reg, l2_reg,
                         link="linear", non_negative=True):
        # TODO: Only pass necessary parameters
        return _newton_update_U(self, U, V, X, alpha, l1_reg, l2_reg,
                                link, non_negative)
    
    def _newton_update_V(self, V, U, Z, X, Y, alpha, l1_reg, l2_reg,
                         x_link="linear", y_link="linear", non_negative=True):
        return _newton_update_V(self, V, U, Z, X, Y, alpha, l1_reg, l2_reg,
                               x_link, y_link, non_negative)
        
    def _newton_update_Z(self, Z, V, Y, alpha, l1_reg, l2_reg,
                         link="linear", non_negative=True):
        return _newton_update_Z(self, Z, V, Y, alpha, l1_reg, l2_reg,
                                link, non_negative)
        
    def update_step(self, X, Y, U, V, Z, l1_reg, l2_reg, alpha):
        self._newton_update_U(U, V, X, alpha, l1_reg, l2_reg,
                              non_negative=self.U_non_negative,
                              link=self.x_link)
        self._newton_update_Z(Z, V, Y, alpha, l1_reg, l2_reg,
                              non_negative=self.Z_non_negative,
                              link=self.y_link)
        self._newton_update_V(V, U, Z, X, Y, alpha, l1_reg, l2_reg,
                              non_negative=self.V_non_negative,
                              x_link=self.x_link, y_link=self.y_link)
