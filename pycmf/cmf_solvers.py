import numbers
import time
from random import random

import numpy as np
import scipy
from scipy.sparse import issparse
from scipy.special import expit
from sklearn.decomposition.nmf import _beta_divergence, _beta_loss_to_float
from sklearn.utils.extmath import safe_sparse_dot

USE_CYTHON = False  # currently, cython is disabled due to unsolved numerical bugs
EPSILON = np.finfo(np.float32).eps

INTEGER_TYPES = (numbers.Integral, np.integer)


# utility functions
def sigmoid(M):
    return 1 / (1 + np.exp(- M))


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
    if target is None:
        return 0
    elif link == "linear":
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
                 l1_reg=0, l2_reg=0, alpha=0.5, verbose=0,
                 U_non_negative=True, V_non_negative=True, Z_non_negative=True,
                 update_U=True, update_V=True, update_Z=True,
                 x_link="linear", y_link="linear", hessian_pertubation=0.2,
                 sg_sample_ratio=1., random_state=None):
        self.max_iter = max_iter
        self.tol = tol
        self.beta_loss = _beta_loss_to_float(beta_loss)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.verbose = verbose
        self.U_non_negative = U_non_negative
        self.V_non_negative = V_non_negative
        self.Z_non_negative = Z_non_negative
        self.update_U = update_U
        self.update_V = update_V
        self.update_Z = update_Z
        self.x_link = x_link
        self.y_link = y_link
        self.hessian_pertubation = hessian_pertubation
        self.sg_sample_ratio = sg_sample_ratio
        if random_state is not None:
            np.random.seed(random_state)

    def update_step(self, X, Y, U, V, Z, l1_reg, l2_reg, alpha):
        """A single update step for all the matrices in the factorization."""
        raise NotImplementedError("Implement in concrete subclass to use")

    def compute_error(self, X, Y, U, V, Z):
        return self.alpha * compute_factorization_error(X, U, V.T, self.x_link, self.beta_loss) + \
            (1 - self.alpha) * compute_factorization_error(Y, V, Z.T, self.y_link, self.beta_loss)

    def fit_iterative_update(self, X, Y, U, V, Z):
        """Compute CMF with iterative methods.
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

        previous_error = error_at_init = self.compute_error(X, Y, U, V, Z)

        for n_iter in range(1, self.max_iter + 1):

            self.update_step(X, Y, U, V, Z, self.l1_reg, self.l2_reg, self.alpha)

            # test convergence criterion every 10 iterations
            if self.tol > 0 and n_iter % 10 == 0:
                error = self.compute_error(X, Y, U, V, Z)

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

        if self.update_V:
            delta_V = self._multiplicative_update_v(X, Y, U, V, Z, self.beta_loss, l1_reg,
                                                    l2_reg, gamma)
            V *= delta_V

        if self.update_U:
            delta_U = self._multiplicative_update_u(X, U, V, self.beta_loss, l1_reg, l2_reg, gamma)
            U *= delta_U

        if self.update_Z:
            delta_Z = self._multiplicative_update_z(Y, V, Z, self.beta_loss, l1_reg, l2_reg, gamma)
            Z *= delta_Z


if USE_CYTHON:
    class NewtonSolver(_IterativeCMFSolver):
        """Internal solver that solves using the Newton-Raphson method.
        Updates each row independently using a Newton-Raphson step. Can handle various link functions and settings.
        The gradient and Hessian are computed based on the residual between the target and the estimate.
        Computing the entire target/estimate can be memory intensive, so the option to compute the residual
        based on a stochastic sample can be enabled by setting sg_sample_ratio < 1.0.

        References
        ----------
        Singh, A. P., & Gordon, G. J. (2008). Relational learning via collective matrix factorization.
        Proceeding of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
        KDD 08, 650. https://doi.org/10.1145/1401890.1401969
        """

        def fit_iterative_update(self, X, Y, U, V, Z):
            # handle memory ordering and format issues for speed up
            X_ = X.tocsr() if issparse(X) else np.ascontiguousarray(X) if X is not None else X
            # instead of solving for Y = VZ^T, in order to make access to V continuous
            # for approximating both X and Y, we will solve Y^T = ZV^T
            Y_ = Y.T.tocsr() if issparse(Y) else np.ascontiguousarray(Y.T) if Y is not None else Y

            # U, V, Z must be C-ordered for cython dot product to work
            U = np.ascontiguousarray(U)
            V = np.ascontiguousarray(V)
            Z = np.ascontiguousarray(Z)
            return super().fit_iterative_update(X_, Y_, U, V, Z)

        def update_step(self, X, Y, U, V, Z, l1_reg, l2_reg, alpha):
            if self.update_U:
                _newton_update_left(U, V, X, alpha, l1_reg, l2_reg,
                                    self.x_link, self.U_non_negative,
                                    self.sg_sample_ratio,
                                    self.hessian_pertubation)

            if self.update_Z:
                _newton_update_left(Z, V, Y, 1 - alpha, l1_reg, l2_reg,
                                    self.y_link, self.Z_non_negative,
                                    self.sg_sample_ratio,
                                    self.hessian_pertubation)

            if self.update_V:
                _newton_update_V(V, U, Z, X, Y, alpha, l1_reg, l2_reg,
                                 self.x_link, self.y_link,
                                 self.V_non_negative,
                                 self.sg_sample_ratio,
                                 self.hessian_pertubation)

        def compute_error(self, X, Y, U, V, Z):
            # override because we are solving for Y^T = ZV^T
            return self.alpha * compute_factorization_error(X, U, V.T, self.x_link, self.beta_loss) + \
                (1 - self.alpha) * compute_factorization_error(Y, Z, V.T, self.y_link, self.beta_loss)

else:
    class NewtonSolver(_IterativeCMFSolver):
        """Default implementation when Cython cannot be used."""

        @classmethod
        def _row_newton_update(cls, M, idx, dM, ddM_inv,
                               eta=1.0, non_negative=True):
            M[idx, :] = M[idx, :] - eta * np.dot(dM, ddM_inv)
            if non_negative:
                M[idx, :][M[idx, :] < 0] = 0.

        def _stochastic_sample(self, features, target, axis=0):
            assert(features.shape[axis] == target.shape[axis])
            if self.sg_sample_ratio < 1.:
                sample_size = int(features.shape[axis] * self.sg_sample_ratio)
                sample_mask = np.random.permutation(np.arange(features.shape[axis]))[:sample_size]
                if axis == 0:
                    features_sampled = features[sample_mask, :]
                    target_sampled = target[sample_mask, :]
                elif axis == 1:
                    features_sampled = features[:, sample_mask]
                    target_sampled = target[:, sample_mask]
                else:
                    raise ValueError("Axis {} out of bounds".format(axis))
            else:
                features_sampled = features
                target_sampled = target
            return features_sampled, target_sampled

        def _safe_invert(self, M):
            """Computed according to reccomendations of
            http://web.stanford.edu/class/cme304/docs/newton-type-methods.pdf"""
            if scipy.sparse.issparse(M):
                eigs, V = scipy.sparse.linalg.eigsh(M)
            else:
                eigs, V = scipy.linalg.eigh(M)
                # perturb hessian to be positive definite
                eigs = np.abs(eigs)
                eigs[eigs < self.hessian_pertubation] = self.hessian_pertubation
            return np.dot(np.dot(V, np.diag(1 / eigs)), V.T)

        def _force_flatten(self, v):
            """Forcibly flattens an indexed row or column of a matrix or sparse matrix"""
            if np.ndim(v) > 1:
                if issparse(v):
                    v_ = v.toarray()
                elif isinstance(v, np.matrix):
                    v_ = np.asarray(v)
                else:
                    raise ValueError("Indexing array returns {} dimensions " +
                                     "but is not sparse or a matrix".format(np.ndim(v)))
                return v_.flatten()
            else:
                return v.flatten()

        def _residual(self, left, right, target, link):
            """Computes residual:
                inverse(left @ right, link) - target
            The number of dimensions of the residual and estimate will be the same.
            This is necessary because the indexing behavior of np.ndarrays and scipy sparse matrices are different.
            Specifically, slicing scipy sparse matrices does not return a 1 dimensional vector.
            e.g.
                >>> import numpy as np; from scipy.sparse import csc_matrix
                >>> A = np.array([[1, 2, 3], [4, 5, 6]])
                >>> B = csc_matrix(A)
                >>> A[:, 0].shape
                (2,)
                >>> B[:, 0].shape
                (2, 1)
            """
            estimate = inverse(np.dot(left, right), link)
            ground_truth = target
            if issparse(target) and np.ndim(estimate) == 1:
                return estimate - ground_truth.toarray().flatten()
            else:
                return estimate - ground_truth

        def _armijo(self, x, u, v, i, grad, link, eta=1, c=0.001, tau=0.5, threshold=2 ** (-5)):
            """Return best newton step size.

            Set t=-c m and iteration counter j = 0.
            Until the condition is satisfied that f(x) - f(x + alpha_j * p) >= alpha_j * t repeatedly increment j and set alpha_j = tau * alpha_{j - 1}
            Return alpha_j as solution.

            :param x: row to be decomposed
            :param u: row to be updated
            :param v: fixed row
            """
            current_error = compute_factorization_error(x, u, v.T, link, self.beta_loss)

            t = c * np.dot(grad, - grad)
            not_found = True
            while not_found:
                updated_u = u
                updated_u[i, :] = updated_u[i, :] + eta * grad

                candidate_error = compute_factorization_error(x, updated_u, v.T, link, self.beta_loss)
                if current_error - candidate_error >= eta * t:
                    not_found = False
                else:
                    eta *= tau

                if (eta * grad <= threshold).all():
                    eta = 0
                    not_found = False

            return eta

        def _v_armijo(self, u, v, z, x, y, i, grad, x_link, y_link, eta=1, c=0.001, tau=0.5, threshold=2 ** (-5)):
            current_error = self.alpha * compute_factorization_error(x, u, v.T, x_link, self.beta_loss) + \
                (1 - self.alpha) * compute_factorization_error(y, z, v.T, y_link, self.beta_loss)

            t = c * np.dot(grad, - grad)
            not_found = True
            while not_found:
                updated_v = v
                updated_v[i, :] = updated_v[i, :] + eta * grad

                candidate_error = self.alpha * compute_factorization_error(x, u, updated_v.T, x_link, self.beta_loss) + \
                    (1 - self.alpha) * compute_factorization_error(y, z, updated_v.T, y_link, self.beta_loss)
                if current_error - candidate_error >= eta * t:
                    not_found = False
                else:
                    eta *= tau

                if (eta * grad <= threshold).all():
                    eta = 0
                    not_found = False

            return eta

        def _newton_update_U(self, U, V, X, alpha, l1_reg, l2_reg,
                             link="linear", non_negative=True):
            precompute_dU = self.sg_sample_ratio == 1.
            if precompute_dU:
                # dU is constant across samples
                res_X = inverse(np.dot(U, V.T), link) - X
                dU_full = alpha * np.dot(res_X, V) + l1_reg * np.sign(U) + l2_reg * U
                if issparse(dU_full):
                    dU_full = dU_full.toarray()
                elif isinstance(dU_full, np.matrix):
                    dU_full = np.asarray(dU_full)

            # iterate over rows
            precompute_ddU_inv = (link == "linear" and self.sg_sample_ratio == 1.)
            if precompute_ddU_inv:
                # ddU_inv is constant across samples
                ddU_inv = self._safe_invert(alpha * np.dot(V.T, V) + l2_reg * np.eye(U.shape[1]))

            for i in range(U.shape[0]):
                u_i = U[i, :]
                V_T_sampled, X_sampled = self._stochastic_sample(V.T, X, axis=1)
                if precompute_dU:
                    dU = dU_full[i, :]
                    assert(np.ndim(dU) == 1)
                else:
                    res_X = self._residual(u_i, V_T_sampled, X_sampled[i, :], link)
                    dU = alpha * np.dot(res_X, V_T_sampled.T) + l1_reg * np.sign(u_i) + l2_reg * u_i

                if not precompute_ddU_inv:
                    if link == "linear":
                        ddU_inv = self._safe_invert(alpha * np.dot(V_T_sampled, V_T_sampled.T) +
                                                    l2_reg * np.eye(U.shape[1]))
                    elif link == "logit":
                        D = np.diag(d_sigmoid(np.dot(u_i, V_T_sampled)))
                        ddU_inv = self._safe_invert(alpha * np.dot(np.dot(V_T_sampled, D), V_T_sampled.T))

                # Calculate armijo and set eta
                eta = self._armijo(X, U, V, i, - np.dot(dU, ddU_inv), link)

                self._row_newton_update(U, i, dU, ddU_inv, non_negative=non_negative, eta=eta)

        def _newton_update_V(self, V, U, Z, X, Y, alpha, l1_reg, l2_reg,
                             x_link="linear", y_link="linear", non_negative=True):
            precompute_dV = (self.sg_sample_ratio == 1.)
            if precompute_dV:
                res_X_T = inverse(np.dot(U, V.T), x_link) - X
                res_Y_T = inverse(np.dot(Z, V.T), y_link) - Y.T
                dV_full = alpha * np.dot(res_X_T.T, U) + \
                    (1 - alpha) * np.dot(res_Y_T.T, Z) + \
                    l1_reg * np.sign(V) + l2_reg * V

                if isinstance(dV_full, np.matrix):
                    dV_full = np.asarray(dV_full)

            precompute_ddV_inv = (x_link == "linear" and y_link == "linear" and self.sg_sample_ratio == 1.)
            if precompute_ddV_inv:
                # ddV_inv is constant w.r.t. the samples of V, so we precompute it to save computation
                ddV_inv = self._safe_invert(alpha * np.dot(U.T, U) +
                                            (1 - alpha) * np.dot(Z.T, Z) +
                                            l2_reg * np.eye(V.shape[1]))

            for i in range(V.shape[0]):
                v_i = V[i, :]

                U_sampled, X_sampled = self._stochastic_sample(U, X)
                Z_T_sampled, Y_sampled = self._stochastic_sample(Z.T, Y, axis=1)

                if not precompute_dV:
                    res_X = self._residual(U_sampled, v_i.T, X_sampled[:, i], x_link)
                    res_Y = self._residual(v_i, Z_T_sampled, Y_sampled[i, :], y_link)
                    dV = alpha * np.dot(res_X.T, U_sampled) + \
                        (1 - alpha) * np.dot(res_Y, Z_T_sampled.T) + \
                        l1_reg * np.sign(v_i) + l2_reg * v_i
                else:
                    dV = dV_full[i, :]

                if not precompute_ddV_inv:
                    if x_link == "logit":
                        D_u = np.diag(d_sigmoid(np.dot(U_sampled, v_i.T)))
                        ddV_wrt_U = np.dot(np.dot(U_sampled.T, D_u), U_sampled)
                    elif x_link == "linear":
                        ddV_wrt_U = np.dot(U_sampled.T, U_sampled)

                    if y_link == "logit":
                        # in the original paper, the equation was v_i.T @ Z,
                        # which clearly does not work due to the dimensionality
                        D_z = np.diag(d_sigmoid(np.dot(v_i, Z_T_sampled)))
                        ddV_wrt_Z = np.dot(np.dot(Z_T_sampled, D_z), Z_T_sampled.T)
                    elif y_link == "linear":
                        ddV_wrt_Z = np.dot(Z_T_sampled, Z_T_sampled.T)

                    ddV_inv = self._safe_invert(alpha * ddV_wrt_U +
                                                (1 - alpha) * ddV_wrt_Z +
                                                l2_reg * np.eye(V.shape[1]))

                eta = self._v_armijo(U, V, Z, X, Y, i, - np.dot(dV, ddV_inv), x_link, y_link)

                self._row_newton_update(V, i, dV, ddV_inv, non_negative=non_negative)

        def _newton_update_Z(self, Z, V, Y, alpha, l1_reg, l2_reg,
                             link="linear", non_negative=True):

            for i in range(Z.shape[0]):
                z_i = Z[i, :]

                V_sampled, Y_sampled = self._stochastic_sample(V, Y)
                res_Y = self._residual(V_sampled, z_i.T, Y_sampled[:, i], link)

                dZ = (1 - alpha) * np.dot(res_Y.T, V_sampled) + \
                    l1_reg * np.sign(z_i) + l2_reg * z_i

                if link == "linear":
                    ddZ_inv = self._safe_invert((1 - alpha) * np.dot(V_sampled.T, V_sampled) +
                                                l2_reg * np.eye(Z.shape[1]))
                elif link == "logit":
                    D = np.diag(d_sigmoid(np.dot(V_sampled, z_i.T)))
                    ddZ_inv = self._safe_invert((1 - alpha) * np.dot(np.dot(V_sampled.T, D), V_sampled) +
                                                l2_reg * np.eye(Z.shape[1]))

                # Calculate armijo and set eta
                eta = self._armijo(Y, Z, V, i, - np.dot(dZ, ddZ_inv), link)

                self._row_newton_update(Z, i, dZ, ddZ_inv, non_negative=non_negative, eta=eta)

        def update_step(self, X, Y, U, V, Z, l1_reg, l2_reg, alpha):
            if self.update_U:
                self._newton_update_U(U, V, X, alpha, l1_reg, l2_reg,
                                      non_negative=self.U_non_negative, link=self.x_link)
            if self.update_Z:
                self._newton_update_Z(Z, V, Y, alpha, l1_reg, l2_reg,
                                      non_negative=self.Z_non_negative, link=self.y_link)

            if self.update_V:
                self._newton_update_V(V, U, Z, X, Y, alpha, l1_reg, l2_reg,
                                      non_negative=self.V_non_negative, x_link=self.x_link,
                                      y_link=self.y_link)
