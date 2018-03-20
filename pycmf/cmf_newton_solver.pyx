# cython: linetrace=True
# cython: profile=True

cimport numpy as np
cimport cython
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
import scipy

EPSILON = np.finfo(np.float32).eps
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# utility functions
@cython.binding(True)
def sigmoid(M):
    return expit(M)

@cython.binding(True)
def d_sigmoid(M):
    sgm = sigmoid(M)
    return sgm * (1 - sgm)

@cython.binding(True)
def inverse(x, link):
    if link == "linear":
        return x
    elif link == "logit":
        return sigmoid(x)
    else:
        raise ValueError("Invalid link function {}".format(link))

@cython.binding(True)
cdef void _row_newton_update_fast(np.ndarray[DTYPE_t, ndim=2] M,
                                  Py_ssize_t idx,
                                  np.ndarray[DTYPE_t, ndim=1] dM,
                                  np.ndarray[DTYPE_t, ndim=2] ddM_inv,
                                  double eta, bint non_negative):
    M[idx, :] = M[idx, :] - eta * np.dot(dM, ddM_inv)
    if non_negative:
        M[idx, :][M[idx, :] < 0] = 0.

@cython.binding(True)
def _stochastic_sample(np.ndarray[DTYPE_t, ndim=2] features,
                       target, double ratio, int axis):
    cdef int sample_size

    assert(features.shape[axis] == target.shape[axis])
    if ratio < 1.:
        sample_size = int(features.shape[axis] * ratio)
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

@cython.binding(True)
def _safe_invert(np.ndarray[DTYPE_t, ndim=2] M, double hessian_pertubation):
    """Computed according to reccomendations of
    http://web.stanford.edu/class/cme304/docs/newton-type-methods.pdf"""
    if issparse(M):
        eigs, V = scipy.sparse.linalg.eigsh(M)
    else:
        eigs, V = scipy.linalg.eigh(M)
        # perturb hessian to be positive definite
        eigs = np.abs(eigs)
        eigs[eigs < hessian_pertubation] = hessian_pertubation
    return np.dot(np.dot(V, np.diag(1 / eigs)), V.T)

# substitutions for np.dot written in pure Cython
import scipy
import scipy.linalg.blas as fblas
from cpython cimport PyCapsule_GetPointer # PyCObject_AsVoidPtr

# check for fortan here:
cdef extern from "numpy/arrayobject.h":
    cdef bint PyArray_IS_F_CONTIGUOUS(np.ndarray) nogil

ctypedef void (*sgemm_ptr) (char *transA, char *transB,
                            int *m, int *n, int *k,\
                            float *alpha,\
                            float *a, int *lda,\
                            float *b, int *ldb,\
                            float *beta, \
                            float *c, int *ldc)

ctypedef void (*dgemm_ptr) (char *transA, char *transB, \
                            int *m, int *n, int *k,\
                            double *alpha,\
                            double *a, int *lda,\
                            double *b, int *ldb,\
                            double *beta, \
                            double *c, int *ldc)

cdef sgemm_ptr sgemm=<sgemm_ptr>PyCapsule_GetPointer(fblas.sgemm._cpointer, NULL)
cdef dgemm_ptr dgemm=<dgemm_ptr>PyCapsule_GetPointer(fblas.dgemm._cpointer, NULL)

@cython.binding(True)
@cython.boundscheck(False)
def matmul(np.ndarray[DTYPE_t, ndim=2] _a, np.ndarray[DTYPE_t, ndim=2] _b,
        bint transA=False, bint transB=False,
        DTYPE_t alpha=1., DTYPE_t beta=0.):
    """Based on https://gist.github.com/JonathanRaiman/07046b897709fffb49e5"""
    cdef int m, n, k, lda, ldb, ldc
    assert(PyArray_IS_F_CONTIGUOUS(_a))
    assert(PyArray_IS_F_CONTIGUOUS(_b))
    cdef char * transA_char = "T" if transA else "N"
    cdef char * transB_char = "T" if transB else "N"
    cdef DTYPE_t * a
    cdef DTYPE_t * b
    cdef DTYPE_t * c

    if transA:
        m = _a.shape[1]
        k = _a.shape[0] # k is the shared dimension
    else:
        m = _a.shape[0]
        k = _a.shape[1]
    n = _b.shape[0 if transB else 1]

    cdef np.ndarray[DTYPE_t, ndim=2] _c = np.zeros((m,n), dtype=DTYPE, order="F")

    a = <DTYPE_t *>np.PyArray_DATA(_a)
    b = <DTYPE_t *>np.PyArray_DATA(_b)
    c = <DTYPE_t *>np.PyArray_DATA(_c)
    lda = _a.shape[0]
    ldb = _b.shape[0]
    ldc = _c.shape[0]
    dgemm(transA_char, transB_char, &m, &n, &k, &alpha, &a[0], &lda, &b[0], &ldb,
          &beta, &c[0], &ldc)
    return _c


@cython.binding(True)
def _residual(np.ndarray[DTYPE_t, ndim=1] left,
              np.ndarray[DTYPE_t, ndim=2] right,
              target, str link, bint should_flatten):
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
    cdef Py_ssize_t i
    cdef double v
    cdef bint is_csr

    estimate = matmul(left.reshape((1, -1)), right).flatten()
    estimate = inverse(estimate, link)
    if should_flatten:
        # return estimate - target.toarray().flatten()
        if not target.getformat() == "csr":
            target = target.tocsr()
        # we assume that sp_arr is a 1 x d matrix
        # this allows us to ignore the row pointers
        for (i, v) in zip(target.indices, target.data):
            estimate[i] -= v
        return estimate
    else:
        return estimate - target

@cython.binding(True)
def _newton_update_left(np.ndarray[DTYPE_t, ndim=2] U,
                        np.ndarray[DTYPE_t, ndim=2] V,
                        X, double alpha,
                        double l1_reg, double l2_reg,
                        str link, bint non_negative,
                        double sg_sample_ratio,
                        double hessian_pertubation):
    cdef int i

    precompute_dU = sg_sample_ratio == 1.
    if precompute_dU:
        # dU is constant across samples
        res_X = inverse(np.dot(U, V.T), link) - X
        dU_full = alpha * np.dot(res_X, V) + l1_reg * np.sign(U) + l2_reg * U
        if issparse(dU_full):
            dU_full = dU_full.toarray()
        elif isinstance(dU_full, np.matrix):
            dU_full = np.asarray(dU_full)

    # iterate over rows
    precompute_ddU_inv = (link == "linear" and sg_sample_ratio == 1.)
    if precompute_ddU_inv:
        # ddU_inv is constant across samples
        ddU_inv = _safe_invert(alpha * np.dot(V.T, V) +
                               l2_reg * np.eye(U.shape[1]),
                               hessian_pertubation)

    res_X_should_flatten = issparse(X)
    for i in range(U.shape[0]):
        u_i = U[i, :]
        # TODO: Fix sampling procedure to take memory order into account and be much faster
        # That being said, sampling column-wise for X is super slow, so some precomputation might be necessary
        V_T_sampled, X_sampled = _stochastic_sample(V.T, X, sg_sample_ratio, 1)
        if precompute_dU:
            dU = dU_full[i, :]
        else:
            res_X = _residual(u_i, V_T_sampled, X_sampled[i, :], link, res_X_should_flatten)
            dU = alpha * np.dot(res_X, V_T_sampled.T) + l1_reg * np.sign(u_i) + l2_reg * u_i

        if not precompute_ddU_inv:
            if link == "linear":
                ddU_inv = _safe_invert(alpha * np.dot(V_T_sampled, V_T_sampled.T) +
                                       l2_reg * np.eye(U.shape[1]),
                                       hessian_pertubation)
            elif link == "logit":
                D = np.diag(d_sigmoid(np.dot(u_i, V_T_sampled)))
                ddU_inv = _safe_invert(alpha * np.dot(np.dot(V_T_sampled, D), V_T_sampled.T),
                                       hessian_pertubation)

        _row_newton_update_fast(U, i, dU, ddU_inv, 1., non_negative)


@cython.binding(True)
def _newton_update_V(np.ndarray[DTYPE_t, ndim=2] V,
                     np.ndarray[DTYPE_t, ndim=2] U,
                     np.ndarray[DTYPE_t, ndim=2] Z,
                     X, Y, double alpha, double l1_reg,
                     double l2_reg, str x_link,
                     str y_link, bint non_negative,
                     double sg_sample_ratio,
                     double hessian_pertubation):
    cdef int i

    precompute_ddV_inv = (x_link == "linear" and y_link == "linear" and sg_sample_ratio == 1.)
    if precompute_ddV_inv:
        # ddV_inv is constant w.r.t. the samples of V, so we precompute it to save computation
        ddV_inv = _safe_invert(alpha * np.dot(U.T, U) +
                               (1 - alpha) * np.dot(Z.T, Z) +
                               l2_reg * np.eye(V.shape[1]),
                               hessian_pertubation)
    res_X_should_flatten = issparse(X)
    res_Y_should_flatten = issparse(Y)
    for i in range(V.shape[0]):
        v_i = V[i, :]

        U_sampled, X_sampled = _stochastic_sample(U, X, sg_sample_ratio, 0)
        res_X_T = _residual(v_i, U_sampled.T, X_sampled[:, i], x_link, res_X_should_flatten)

        Z_sampled, Y_sampled = _stochastic_sample(Z, Y, sg_sample_ratio, 0)
        res_Y_T = _residual(v_i, Z_sampled.T, Y_sampled[:, i], y_link, res_Y_should_flatten)

        dV = alpha * np.dot(res_X_T, U_sampled) + \
            (1 - alpha) * np.dot(res_Y_T, Z_sampled) + \
            l1_reg * np.sign(v_i) + l2_reg * v_i

        if not precompute_ddV_inv:
            if x_link == "logit":
                D_u = np.diag(d_sigmoid(np.dot(U_sampled, v_i.T)))
                ddV_wrt_U = np.dot(np.dot(U_sampled.T, D_u), U_sampled)
            elif x_link == "linear":
                ddV_wrt_U = np.dot(U_sampled.T, U_sampled)

            if y_link == "logit":
                # in the original paper, the equation was v_i.T @ Z,
                # which clearly does not work due to the dimensionality
                D_z = np.diag(d_sigmoid(np.dot(Z_sampled, v_i.T)))
                ddV_wrt_Z = np.dot(np.dot(Z_sampled.T, D_z), Z_sampled)
            elif y_link == "linear":
                ddV_wrt_Z = np.dot(Z_sampled.T, Z_sampled)

            ddV_inv = _safe_invert(alpha * ddV_wrt_U +
                                   (1 - alpha) * ddV_wrt_Z +
                                   l2_reg * np.eye(V.shape[1]),
                                   hessian_pertubation)

        _row_newton_update_fast(V, i, dV, ddV_inv, 1., non_negative)
