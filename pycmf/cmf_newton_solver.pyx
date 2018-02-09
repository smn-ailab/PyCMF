# cython: linetrace=True

cimport numpy as np
cimport cython
import numpy as npp
from scipy.sparse import issparse
from scipy.special import expit
import scipy

EPSILON = npp.finfo(npp.float32).eps

# utility functions
@cython.profile(False)
def sigmoid(M):
    # TODO: Speed up using precalculation
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

def _row_newton_update_fast(np.ndarray[double, ndim=2] M,
                                  Py_ssize_t idx,
                                  np.ndarray[double, ndim=1] dM,
                                  np.ndarray[double, ndim=2] ddM_inv,
                                  double eta, bint non_negative):
    M[idx, :] = M[idx, :] - eta * npp.dot(dM, ddM_inv)
    if non_negative:
        M[idx, :][M[idx, :] < 0] = 0.

def _stochastic_sample(np.ndarray[double, ndim=2] features,
                       target, double ratio, int axis):
    cdef int sample_size
    
    assert(features.shape[axis] == target.shape[axis])
    if ratio < 1.:
        sample_size = int(features.shape[axis] * ratio)
        sample_mask = npp.random.permutation(npp.arange(features.shape[axis]))[:sample_size]
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

def _safe_invert(np.ndarray[double, ndim=2] M, double hessian_pertubation):
    """Computed according to reccomendations of
    http://web.stanford.edu/class/cme304/docs/newton-type-methods.pdf"""
    if issparse(M):
        eigs, V = scipy.sparse.linalg.eigsh(M)
    else:
        eigs, V = scipy.linalg.eigh(M)
        # perturb hessian to be positive definite
        eigs = npp.abs(eigs)
        eigs[eigs < hessian_pertubation] = hessian_pertubation
    return npp.dot(npp.dot(V, npp.diag(1 / eigs)), V.T)
    
def _residual(np.ndarray[double, ndim=1] left,
              np.ndarray[double, ndim=2] right,
              target, str link):
    """Computes residual:
        inverse(left @ right, link) - target
    The number of dimensions of the residual and estimate will be the same.
    This is necessary because the indexing behavior of npp.ndarrays and scipy sparse matrices are different.
    Specifically, slicing scipy sparse matrices does not return a 1 dimensional vector.
    e.g.
        >>> import numpy as npp; from scipy.sparse import csc_matrix
        >>> A = npp.array([[1, 2, 3], [4, 5, 6]])
        >>> B = csc_matrix(A) 
        >>> A[:, 0].shape
        (2,)
        >>> B[:, 0].shape
        (2, 1)
    """
    estimate = inverse(npp.dot(left, right), link)
    ground_truth = target
    if issparse(target) and npp.ndim(estimate) == 1:
        return estimate - ground_truth.toarray().flatten()
    else:
        return estimate - ground_truth
    
def _newton_update_U(self, np.ndarray[double, ndim=2] U,
                     np.ndarray[double, ndim=2] V,
                     X, double alpha,
                     double l1_reg, double l2_reg,
                     str link, bint non_negative):
    cdef int i
    
    precompute_dU = self.sg_sample_ratio == 1.
    if precompute_dU:
        # dU is constant across samples
        res_X = inverse(npp.dot(U, V.T), link) - X
        dU_full = alpha * npp.dot(res_X, V) + l1_reg * npp.sign(U) + l2_reg * U
        if issparse(dU_full):
            dU_full = dU_full.toarray()
        elif isinstance(dU_full, npp.matrix):
            dU_full = npp.asarray(dU_full)

    # iterate over rows
    precompute_ddU_inv = (link == "linear" and self.sg_sample_ratio == 1.)
    if precompute_ddU_inv:
        # ddU_inv is constant across samples
        ddU_inv = _safe_invert(alpha * npp.dot(V.T, V) +
                               l2_reg * npp.eye(U.shape[1]),
                               self.hessian_pertubation)

    for i in range(U.shape[0]):
        u_i = U[i, :]
        V_T_sampled, X_sampled = _stochastic_sample(V.T, X, self.sg_sample_ratio, 1)
        if precompute_dU:
            dU = dU_full[i, :]
            assert(npp.ndim(dU) == 1)
        else:
            res_X = _residual(u_i, V_T_sampled, X_sampled[i, :], link)
            dU = alpha * npp.dot(res_X, V_T_sampled.T) + l1_reg * npp.sign(u_i) + l2_reg * u_i

        if not precompute_ddU_inv:
            if link == "linear":
                ddU_inv = _safe_invert(alpha * npp.dot(V_T_sampled, V_T_sampled.T) +
                                       l2_reg * npp.eye(U.shape[1]),
                                       self.hessian_pertubation)
            elif link == "logit":
                D = npp.diag(d_sigmoid(npp.dot(u_i, V_T_sampled)))
                ddU_inv = _safe_invert(alpha * npp.dot(npp.dot(V_T_sampled, D), V_T_sampled.T),
                                       self.hessian_pertubation)

        _row_newton_update_fast(U, i, dU, ddU_inv, 1., non_negative)
        
def _newton_update_V(self, np.ndarray[double, ndim=2] V,
                     np.ndarray[double, ndim=2] U, np.ndarray[double, ndim=2] Z,
                     X, Y, double alpha, double l1_reg,
                     double l2_reg, str x_link,
                     str y_link, bint non_negative):
    cdef int i
    
    precompute_ddV_inv = (x_link == "linear" and y_link == "linear" and self.sg_sample_ratio == 1.)
    if precompute_ddV_inv:
        # ddV_inv is constant w.r.t. the samples of V, so we precompute it to save computation
        ddV_inv = _safe_invert(alpha * npp.dot(U.T, U) + 
                               (1 - alpha) * npp.dot(Z.T, Z) +
                               l2_reg * npp.eye(V.shape[1]),
                               self.hessian_pertubation)

    for i in range(V.shape[0]):
        v_i = V[i, :]

        U_sampled, X_sampled = _stochastic_sample(U, X, self.sg_sample_ratio, 0)
        # res_X = _residual(U_sampled, v_i.T, X_sampled[:, i], x_link)
        res_X = _residual(v_i, U_sampled.T, X_sampled[:, i].T, x_link).T

        Z_T_sampled, Y_sampled = _stochastic_sample(Z.T, Y, self.sg_sample_ratio, 1)
        res_Y = _residual(v_i, Z_T_sampled, Y_sampled[i, :], y_link)

        dV = alpha * npp.dot(res_X.T, U_sampled) + \
            (1 - alpha) * npp.dot(res_Y, Z_T_sampled.T) + \
            l1_reg * npp.sign(v_i) + l2_reg * v_i

        if not precompute_ddV_inv:
            if x_link == "logit":
                D_u = npp.diag(d_sigmoid(npp.dot(U_sampled, v_i.T)))
                ddV_wrt_U = npp.dot(npp.dot(U_sampled.T, D_u), U_sampled)
            elif x_link == "linear":
                ddV_wrt_U = npp.dot(U_sampled.T, U_sampled)

            if y_link == "logit":
                # in the original paper, the equation was v_i.T @ Z,
                # which clearly does not work due to the dimensionality
                D_z = npp.diag(d_sigmoid(npp.dot(v_i, Z_T_sampled)))
                ddV_wrt_Z = npp.dot(npp.dot(Z_T_sampled, D_z), Z_T_sampled.T)
            elif y_link == "linear":
                ddV_wrt_Z = npp.dot(Z_T_sampled, Z_T_sampled.T)

            ddV_inv = _safe_invert(alpha * ddV_wrt_U +
                                   (1 - alpha) * ddV_wrt_Z +
                                   l2_reg * npp.eye(V.shape[1]),
                                   self.hessian_pertubation)

        _row_newton_update_fast(V, i, dV, ddV_inv, 1., non_negative)

def _newton_update_Z(self, np.ndarray[double, ndim=2] Z,
                     np.ndarray[double, ndim=2] V,
                     Y, double alpha, double l1_reg,
                     double l2_reg, str link,
                     bint non_negative):
    cdef int i
    
    for i in range(Z.shape[0]):
        z_i = Z[i, :]

        V_sampled, Y_sampled = _stochastic_sample(V, Y, self.sg_sample_ratio, 0)
        # res_Y = _residual(V_sampled, z_i.T, Y_sampled[:, i], link)
        res_Y = _residual(z_i, V_sampled.T, Y_sampled[:, i].T, link).T

        dZ = (1 - alpha) * npp.dot(res_Y.T, V_sampled) + \
            l1_reg * npp.sign(z_i) + l2_reg * z_i

        if link == "linear":
            ddZ_inv = _safe_invert((1 - alpha) * npp.dot(V_sampled.T, V_sampled) +
                                   l2_reg * npp.eye(Z.shape[1]),
                                   self.hessian_pertubation)
        elif link == "logit":
            D = npp.diag(d_sigmoid(npp.dot(V_sampled, z_i.T)))
            ddZ_inv = _safe_invert((1 - alpha) * npp.dot(npp.dot(V_sampled.T, D), V_sampled) +
                                   l2_reg * npp.eye(Z.shape[1]),
                                   self.hessian_pertubation)

        _row_newton_update_fast(Z, i, dZ, ddZ_inv, 1., non_negative)
