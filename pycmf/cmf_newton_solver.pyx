# cython: linetrace=True

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

# check for fortan here:
cdef extern from "numpy/arrayobject.h":
    cdef bint PyArray_IS_F_CONTIGUOUS(np.ndarray) nogil

# with those pointers we can now wrap a cython function for these

def matmul(np.ndarray[DTYPE_t, ndim=2] _a, np.ndarray[DTYPE_t, ndim=2] _b,
        bint transA=False, bint transB=False,
        DTYPE_t alpha=1., DTYPE_t beta=0.):
    """Based on https://gist.github.com/JonathanRaiman/07046b897709fffb49e5"""
    cdef int m, n, k, lda, ldb, ldc
    if not PyArray_IS_F_CONTIGUOUS(_a):
      _a = _a.T
      transA = not transA
    if not PyArray_IS_F_CONTIGUOUS(_b):
      _b = _b.T
      transB = not transB
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

    # some of the operations above
    # are gil needy and thus
    # only this last chunk can be "ungiled"
    # when life give you lemons make lemonade
    lda = _a.shape[0]
    ldb = _b.shape[0]
    ldc = _c.shape[0]
    dgemm(transA_char, transB_char, &m, &n, &k, &alpha, &a[0], &lda, &b[0], &ldb,
          &beta, &c[0], &ldc)
    return _c

# borrowed from some other source
from cpython cimport PyCapsule_GetPointer # PyCObject_AsVoidPtr
from sklearn.utils.validation import check_array as array2d

def _impose_f_order(X):
    """Helper Function"""
    # important to access flags instead of calling np.isfortran,
    # this catches corner cases.
    if X.flags.c_contiguous:
        return array2d(X.T, copy=False, order='F'), True
    else:
        return array2d(X, copy=False, order='F'), False

ctypedef void (*sgemm_ptr) (char *transA, char *transB, \
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
@cython.boundscheck(True)
def vector_matrix_prod(np.ndarray[DTYPE_t, ndim=1] A,
                       np.ndarray[DTYPE_t, ndim=2] B):
    """Cython implementation of fast vector matrix product"""
    # dot = scipy.linalg.get_blas_funcs('gemm', (A, B))
    # A_, trans_a = _impose_f_order(A.reshape(1, -1))
    # B, trans_b = _impose_f_order(B)
    # TODO: Remove this step...?
    return matmul(A.reshape((1, -1)), B)

@cython.binding(True)
@cython.boundscheck(True)
def matrix_vector_prod(np.ndarray[DTYPE_t, ndim=2] matrix,
                       np.ndarray[DTYPE_t, ndim=1] vector):
    """Cython implementation of fast matrix vector product"""
    cdef int k = vector.shape[0]
    cdef int boundary = matrix.shape[0]
    cdef int i, j
    cdef double acc
    cdef np.ndarray[DTYPE_t, ndim=1] out = np.zeros((boundary, 1))
    for i in range(boundary):
        acc = 0.
        for j in range(k):
            acc += vector[j] * matrix[i, j]
        out[i] = acc
    return out

@cython.binding(True)
@cython.boundscheck(True)
def matrix_matrix_prod(np.ndarray[DTYPE_t, ndim=2] matrix,
                       np.ndarray[DTYPE_t, ndim=2] flat_matrix):
    """Matrix multiplication between matrix of shape (d, k) and (k, 1)"""
    cdef int k = flat_matrix.shape[0]
    cdef int d = matrix.shape[0]
    cdef int i, j
    cdef double acc
    cdef np.ndarray[double, ndim=1] out = np.zeros((d, 1))
    for i in range(d):
        acc = 0.
        for j in range(k):
            acc += flat_matrix[j, 0] * matrix[i, j]
        out[i] = acc
    return out

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
    estimate = vector_matrix_prod(left, right).flatten()
    estimate = inverse(estimate, link)
    if should_flatten:
        return estimate - target.toarray().flatten()
    else:
        return estimate - target

@cython.binding(True)
def _residual_T(np.ndarray[DTYPE_t, ndim=2] left,
                np.ndarray[DTYPE_t, ndim=1] right,
                target, str link, bint should_flatten):
    estimate = matrix_vector_prod(left, right)
    estimate = inverse(estimate, link)
    if should_flatten:
        return estimate - target.toarray().flatten()
    else:
        return estimate - target

@cython.binding(True)
def _newton_update_U(np.ndarray[DTYPE_t, ndim=2] U,
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
        res_X = inverse(np.dot(U, V.T), link) - X #SLOW: 35%
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
        V_T_sampled, X_sampled = _stochastic_sample(V.T, X, sg_sample_ratio, 1) #SLOW: 9.4%
        if precompute_dU:
            dU = dU_full[i, :]
            # assert(np.ndim(dU) == 1)
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
        assert(np.ndim(v_i) == 1)
        # res_X = _residual_T(U_sampled, v_i, X_sampled[:, i], x_link, res_X_should_flatten) #SLOW: 50%
        res_X = _residual(v_i, U_sampled.T, X_sampled[:, i].T, x_link, res_X_should_flatten).T

        Z_T_sampled, Y_sampled = _stochastic_sample(Z.T, Y, sg_sample_ratio, 1)
        res_Y = _residual(v_i, Z_T_sampled, Y_sampled[i, :], y_link, res_Y_should_flatten)

        dV = alpha * np.dot(res_X.T, U_sampled) + \
            (1 - alpha) * np.dot(res_Y, Z_T_sampled.T) + \
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
                D_z = np.diag(d_sigmoid(np.dot(v_i, Z_T_sampled)))
                ddV_wrt_Z = np.dot(np.dot(Z_T_sampled, D_z), Z_T_sampled.T)
            elif y_link == "linear":
                ddV_wrt_Z = np.dot(Z_T_sampled, Z_T_sampled.T)

            ddV_inv = _safe_invert(alpha * ddV_wrt_U +
                                   (1 - alpha) * ddV_wrt_Z +
                                   l2_reg * np.eye(V.shape[1]),
                                   hessian_pertubation)

        _row_newton_update_fast(V, i, dV, ddV_inv, 1., non_negative)

@cython.binding(True)
def _newton_update_Z(np.ndarray[DTYPE_t, ndim=2] Z,
                     np.ndarray[DTYPE_t, ndim=2] V,
                     Y, double alpha, double l1_reg,
                     double l2_reg, str link,
                     bint non_negative,
                     double sg_sample_ratio,
                     double hessian_pertubation):
    cdef int i

    res_Y_should_flatten = issparse(Y)
    for i in range(Z.shape[0]):
        z_i = Z[i, :]

        V_sampled, Y_sampled = _stochastic_sample(V, Y, sg_sample_ratio, 0)
        # res_Y = _residual_T(V_sampled, z_i, Y_sampled[:, i], link, res_Y_should_flatten) #SLOW: 6.4%
        res_Y = _residual(z_i, V_sampled.T, Y_sampled[:, i].T, link, res_Y_should_flatten).T

        dZ = (1 - alpha) * np.dot(res_Y.T, V_sampled) + \
            l1_reg * np.sign(z_i) + l2_reg * z_i

        if link == "linear":
            ddZ_inv = _safe_invert((1 - alpha) * np.dot(V_sampled.T, V_sampled) +
                                   l2_reg * np.eye(Z.shape[1]),
                                   hessian_pertubation)
        elif link == "logit":
            D = np.diag(d_sigmoid(np.dot(V_sampled, z_i.T))) #SLOW: 10.2%
            ddZ_inv = _safe_invert((1 - alpha) * np.dot(np.dot(V_sampled.T, D), V_sampled) + #SLOW: 16.%
                                   l2_reg * np.eye(Z.shape[1]),
                                   hessian_pertubation) #SLOW: 50.7%

        _row_newton_update_fast(Z, i, dZ, ddZ_inv, 1., non_negative)
