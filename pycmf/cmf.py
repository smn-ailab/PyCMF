from __future__ import division, print_function
from math import sqrt
import warnings
import numbers
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, squared_norm
from sklearn.utils.validation import check_non_negative

# solvers
from .cmf_solvers import MUSolver, NewtonSolver, compute_factorization_error
from .analysis import _print_topic_terms_from_matrix, _print_topic_terms_with_importances_from_matrices

EPSILON = np.finfo(np.float32).eps

INTEGER_TYPES = (numbers.Integral, np.integer)


# initialization utils

def norm(x):
    """Dot product-based Euclidean norm implementation

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    """
    return sqrt(squared_norm(x))


def _check_init(A, shape, whom, non_negative):
    A = check_array(A)
    if np.shape(A) != shape:
        raise ValueError('Array with wrong shape passed to %s. Expected %s, '
                         'but got %s ' % (whom, shape, np.shape(A)))
    if non_negative:
        check_non_negative(A, whom)
        if np.max(A) == 0:
            raise ValueError('Array passed to %s is full of zeros.' % whom)


def _initialize_mf(M, n_components, init=None, eps=1e-6, random_state=None, non_negative=False):
    """Algorithms for MF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for M: M = AB^T

    Parameters
    ----------
    M : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : integer
        The number of components desired in the approximation.

    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'svd'
        Method used to initialize the procedure.
        Default: 'svd' if n_components < n_features, otherwise 'random'.
        Valid options:

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

    non_negative: bool
        Whether to decompose into non-negative matrices.

    eps : float
        If non-negative, truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``random`` == 'nndsvdar' or 'random'.

    Returns
    -------
    A : array-like, shape (n_samples, n_components)
        Initial guesses for solving M ~= AB^T

    B : array-like, shape (n_features, n_components)
        Initial guesses for solving M ~= AB^T

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    if non_negative:
        check_non_negative(M, "MF initialization")

    n_samples, n_features = M.shape

    if init is None:
        if n_components < n_features:
            init = 'nndsvdar' if non_negative else 'svd'
        else:
            init = 'random'

    if init == 'random':
        avg = np.sqrt(np.abs(M.mean()) / n_components)
        rng = check_random_state(random_state)
        A = avg * rng.randn(n_samples, n_components)
        B = avg * rng.randn(n_components, n_features)
        if non_negative:
            np.abs(A, A)
            np.abs(B, B)

    elif init == 'svd':
        if non_negative:
            raise ValueError('SVD initialization incompatible with NMF (use nndsvd instead)')
        if min(n_samples, n_features) < n_components:
            warnings.warn('The number of components is smaller than the rank in svd initialization.' +
                          'The input will be padded with zeros to compensate for the lack of singular values.')
        # simple SVD based approximation
        U, S, V = randomized_svd(M, n_components, random_state=random_state)
        # randomize_svd only returns min(n_components, n_features, n_samples) singular values and vectors
        # therefore, to retain the desired shape, we need to pad and reshape the inputs
        if n_components > n_features:
            U_padded = np.zeros((U.shape[0], n_components))
            U_padded[:, :U.shape[1]] = U
            U = U_padded
            V_padded = np.zeros((n_components, V.shape[1]))
            V_padded[:V.shape[0], :] = V
            V = V_padded
            S_padded = np.zeros(n_components)
            S_padded[:S.shape[0]] = S
            S = S_padded

        S = np.diag(np.sqrt(S))
        A = np.dot(U, S)
        B = np.dot(S, V)

    elif init in ['nndsvd', 'nndsvda', 'nndsvdar']:
        if not non_negative:
            warnings.warn('%s results in non-negative constrained factors,' % init +
                          'so SVD initialization should provide better initial estimate')
        # NNDSVD initialization
        U, S, V = randomized_svd(M, n_components, random_state=random_state)
        A, B = np.zeros(U.shape), np.zeros(V.shape)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        A[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        B[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        for j in range(1, n_components):
            x, y = U[:, j], V[j, :]

            # extract positive and negative parts of column vectors
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

            # and their norms
            x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
            x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n

            lbd = np.sqrt(S[j] * sigma)
            A[:, j] = lbd * u
            B[j, :] = lbd * v

        A[A < eps] = 0
        B[B < eps] = 0

        if init == "nndsvd":
            pass
        elif init == "nndsvda":
            avg = M.mean()
            A[A == 0] = avg
            B[B == 0] = avg
        elif init == "nndsvdar":
            rng = check_random_state(random_state)
            avg = M.mean()
            A[A == 0] = abs(avg * rng.randn(len(A[A == 0])) / 100)
            B[B == 0] = abs(avg * rng.randn(len(B[B == 0])) / 100)

    else:
        raise ValueError("Invalid init argument")

    return A, B.T


def _init_custom(A, M, n_components, idx,
                 non_negative=False, random_state=None):
    if A is not None:
        _check_init(A, (M.shape[idx], n_components), "CMF (input {})".format(idx), non_negative)
        return A
    else:
        return _initialize_mf(M, n_components, init="random", random_state=random_state,
                              non_negative=non_negative)[idx]


def collective_matrix_factorization(X, Y, U=None, V=None, Z=None,
                                    n_components=None, solver="mu", alpha=0.5,
                                    x_init=None, y_init=None, beta_loss="frobenius",
                                    tol=1e-4, l1_reg=0., l2_reg=0.,
                                    random_state=None, max_iter=200, verbose=0,
                                    U_non_negative=True, V_non_negative=True,
                                    Z_non_negative=True, update_U=True,
                                    update_V=True, update_Z=True,
                                    x_link="linear", y_link="linear",
                                    hessian_pertubation=0.2, sg_sample_ratio=1.):
    """Compute Collective Matrix Factorization (CMF)

    Currently only available for factorizing two matrices X and Y.
    Find low-rank, non-negative matrices (U, V, Z) that can approximate X and Y simultaneously.

    The objective function is::

        alpha * ||X - f_1(UV^T)||_Fro^2
        + (1 - alpha) * ||Y - f_2(VZ^T)||_Fro^2
        + l1_reg * (||U + V + Z||_1)
        + l2_reg * (||U||_2 + ||V||_2 + ||Z||_2)

    Where::

        f_1, f_2: some element-wise functions
        ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
        ||A||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

    For newton solver, f_1 and f_2 can be either the identity function of the logit function.

    For multiplicative-update solver, the Frobenius norm
    (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
    by changing the beta_loss parameter in theory. This is not yet implemented though.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        First data matrix to be decomposed

    Y : {array-like, sparse matrix}, shape (n_features, n_labels)
        Second data matrix to be decomposed. X and Y must satisfy the condition
        X.shape[1] == Y.shape[0] (in other words, XY must be a valid matrix multiplication)

    U : array-like, shape (n_samples, n_components)
        If init='custom', it is used as initial guess for the solution.

    V : array-like, shape (n_features, n_components)
        If init='custom', it is used as initial guess for the solution.

    Z : array-like, shape (n_labels, n_components)
        If init='custom', it is used as initial guess for the solution.

    n_components : integer
        Number of components, if n_components is not set all features
        are kept.

    x_init, y_init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom' | 'svd'
        Method used to initialize the procedure.
        Default: 'nndsvd' if n_components < n_features, otherwise random.
        Valid options:

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices U, V and Z

        - 'svd': use randomized svd to find approximation allowing negative values


    solver : 'newton' | 'mu'
        Numerical solver to use:
        'newton' is the Newton-Raphson solver.
        'mu' is a Multiplicative Update solver.

    beta_loss : float or string, default 'frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    alpha : double, default: 0.5
        Constant that handles balance between the loss for both matrices.

    l1_reg : double, default: 0.
        The regularization parameter for L1 penalty.

    l2_reg : double, default: 0.
        The regularization parameter for L2 penalty.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : integer, default: 0
        The verbosity level.

    U_non_negative: bool, default: True
        Whether to enforce non-negativity for U. Only applicable for the newton solver.

    V_non_negative: bool, default: True
        Whether to enforce non-negativity for V. Only applicable for the newton solver.

    Z_non_negative: bool, default: True
        Whether to enforce non-negativity for Z. Only applicable for the newton solver.

    update_U, update_V, update_Z: bool, default: True
        Whether to update U, V, and Z respectively.

    x_link: str, default: "linear"
        One of either "logit" of "linear". The link function for transforming UV^T to approximate X

    y_link: str, default: "linear"
        One of either "logit" of "linear". The link function for transforming VZ^T to approximate Y

    hessian_pertubation: double, default: 0.2
        The pertubation to the Hessian in the newton solver to maintain positive definiteness

    sg_sample_ratio: double, default: 1.0
        The sample ratio for stochastic gradient in newton solver.
        If 1.0, the gradient is not stochastic.
        Warning: Using sg_sample_ratio < 1.0 can currently be extremely slow.
        It is currently recommended to use sg_sample_ratio = 1.0 whenever possible.

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

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> Y = np.array([[5, 3], [1, 9]])
    >>> from CMF import collective_matrix_factorization
    >>> U, V, Z, n_iter = collective_matrix_factorization(X, Y, n_components=2, \
        init='random', random_state=0)
    """
    if n_components is None:
        n_components = max(X.shape[1], Y.shape[1])

    # sanity check the input
    if update_U or update_V:
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
    if update_Z or update_V:
        Y = check_array(Y, accept_sparse=('csr', 'csc'), dtype=float)

    if update_V:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Expected X.shape[1] == Y.shape[0], " +
                             "found X.shape = {}, Y.shape = {}".format(X.shape[1], Y.shape[0]))

    if x_link not in ["linear", "logit"]:
        raise ValueError("No such link %s for x_link" % x_link)

    if y_link not in ["linear", "logit"]:
        raise ValueError("No such link %s for y_link" % y_link)

    # initialize U, V, and Z
    if x_init == 'custom':
        if X is not None:
            U = _init_custom(U, X, n_components, 0,
                             non_negative=U_non_negative, random_state=random_state)
            V = _init_custom(V, X, n_components, 1,
                             non_negative=V_non_negative, random_state=random_state)
    else:
        x_init = "random" if x_link == "logit" else x_init
        U, V = _initialize_mf(X, n_components, init=x_init, random_state=random_state,
                              non_negative=(U_non_negative or V_non_negative))

    if y_init == 'custom':
        if Y is not None:
            V = _init_custom(V, Y, n_components, 0,
                             non_negative=V_non_negative, random_state=random_state)
            Z = _init_custom(Z, Y, n_components, 1,
                             non_negative=Z_non_negative, random_state=random_state)
        V_ = V
    else:
        y_init = "random" if y_link == "logit" else y_init
        V_, Z = _initialize_mf(Y, n_components, init=y_init, random_state=random_state,
                               non_negative=(Z_non_negative or V_non_negative))

    if U_non_negative == Z_non_negative:
        V = (V + V_) / 2
    elif U_non_negative:
        V = V
    elif Z_non_negative:
        V = V_

    # Solve
    if solver == "mu":
        if x_link != "linear" or y_link != "linear":
            warnings.warn("mu solver does not accept link functions other than linear, link arguments will be ignored")

        solver_object = MUSolver(max_iter=max_iter, tol=tol, verbose=verbose,
                                 update_U=update_U, update_V=update_V, update_Z=update_Z,
                                 l1_reg=l1_reg, l2_reg=l2_reg, beta_loss=beta_loss, random_state=random_state)
    elif solver == "newton":
        if alpha == "auto":
            # adjust alpha so that both X and Y are "equally" considered,
            # i.e. X.shape[0] * alpha = Y.shape[1] * (1 - alpha)
            alpha = Y.shape[1] / (X.shape[0] + Y.shape[1])
        solver_object = NewtonSolver(alpha=alpha, l1_reg=l1_reg, tol=tol,
                                     l2_reg=l2_reg, max_iter=max_iter, verbose=verbose,
                                     update_U=update_U, update_V=update_V, update_Z=update_Z,
                                     U_non_negative=U_non_negative, V_non_negative=V_non_negative,
                                     Z_non_negative=Z_non_negative, x_link=x_link, y_link=y_link,
                                     hessian_pertubation=hessian_pertubation,
                                     sg_sample_ratio=sg_sample_ratio, random_state=random_state)
    else:
        raise ValueError("No such solver: %s" % solver)
    U, V, Z, n_iter = solver_object.fit_iterative_update(X, Y, U, V, Z)

    return U, V, Z, n_iter


class CMF(BaseEstimator, TransformerMixin):
    """Compute Collective Matrix Factorization (CMF)

        Currently only available for factorizing two matrices X and Y.
        Find low-rank, non-negative matrices (U, V, Z) that can approximate X and Y simultaneously.

        The objective function is::

            alpha * ||X - f_1(UV^T)||_Fro^2
            + (1 - alpha) * ||Y - f_2(VZ^T)||_Fro^2
            + l1_reg * (||U + V + Z||_1)
            + l2_reg * (||U||_2 + ||V||_2 + ||Z||_2)

        Where::

            f_1, f_2: some element-wise functions
            ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
            ||A||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

        For newton solver, f_1 and f_2 can be either the identity function of the logit function.

        For multiplicative-update solver, the Frobenius norm
        (0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
        by changing the beta_loss parameter in theory. This is not yet implemented though.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            First data matrix to be decomposed

        Y : {array-like, sparse matrix}, shape (n_features, n_labels)
            Second data matrix to be decomposed. X and Y must satisfy the condition
            X.shape[1] == Y.shape[0] (in other words, XY must be a valid matrix multiplication)

        U : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        V : array-like, shape (n_features, n_components)
            If init='custom', it is used as initial guess for the solution.

        Z : array-like, shape (n_labels, n_components)
            If init='custom', it is used as initial guess for the solution.

        n_components : integer
            Number of components, if n_components is not set all features
            are kept.

        x_init, y_init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom' | 'svd'
            Method used to initialize the procedure.
            Default: 'nndsvd' if n_components < n_features, otherwise random.
            Valid options:

            - 'random': non-negative random matrices, scaled with:
                sqrt(X.mean() / n_components)

            - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
                initialization (better for sparseness)

            - 'nndsvda': NNDSVD with zeros filled with the average of X
                (better when sparsity is not desired)

            - 'nndsvdar': NNDSVD with zeros filled with small random values
                (generally faster, less accurate alternative to NNDSVDa
                for when sparsity is not desired)

            - 'custom': use custom matrices U, V and Z

            - 'svd': use randomized svd to find approximation allowing negative values

        solver : 'newton' | 'mu'
            Numerical solver to use:
            'newton' is the Newton-Raphson solver.
            'mu' is a Multiplicative Update solver.

        alpha: double, default: 'auto'
            Determines trade-off between optimizing for X and Y.
            The larger the value, the more X is prioritized in optimization.
            If set to 'auto', alpha will be computed so that the relative contributions of X and Y will be equivalent.

        beta_loss : float or string, default 'frobenius'
            String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
            Beta divergence to be minimized, measuring the distance between X
            and the dot product WH. Note that values different from 'frobenius'
            (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
            fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
            matrix X cannot contain zeros. Used only in 'mu' solver.

        tol : float, default: 1e-4
            Tolerance of the stopping condition.

        max_iter : integer, default: 200
            Maximum number of iterations before timing out.

        l1_reg : double, default: 0.
            The regularization parameter for L1 penalty.

        l2_reg : double, default: 0.
            The regularization parameter for L2 penalty.

        random_state : int, RandomState instance or None, optional, default: None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        verbose : integer, default: 0
            The verbosity level.

        U_non_negative: bool, default: True
            Whether to enforce non-negativity for U. Only applicable for the newton solver.

        V_non_negative: bool, default: True
            Whether to enforce non-negativity for V. Only applicable for the newton solver.

        Z_non_negative: bool, default: True
            Whether to enforce non-negativity for Z. Only applicable for the newton solver.

        x_link: str, default: "linear"
            One of either "logit" of "linear". The link function for transforming UV^T to approximate X.
            If "linear", UV^T will be used to approximate X.
            If "logit", sigmoid(UV^T) will be used to approximate X.

        y_link: str, default: "linear"
            One of either "logit" of "linear". The link function for transforming VZ^T to approximate Y
            If "linear", VZ^T will be used to approximate Y.
            If "logit", sigmoid(VZ^T) will be used to approximate Y.

        hessian_pertubation: double, default: 0.2
            The pertubation to the Hessian in the newton solver to maintain positive definiteness

        sg_sample_ratio: double, default: 1.0
            The sample ratio for stochastic gradient in newton solver. If 1.0, the gradient is not stochastic.

        Attributes
        ----------
        components : array, [n_features, n_components]
            Factorization matrix V.

        x_weights : array, [n_samples, n_components]
            X components weights U.

        y_weights : array, [n_labels, n_components]
            Y components weights Z.

        reconstruction_err_ : number
            Frobenius norm of the matrix difference, or beta-divergence, between
            the training data ``X``, ``Y`` and the reconstructed data ``UV^T``, ``VZ^T`` from
            the fitted model.

        n_iter_ : int
            Number of iterations that training went on for.

        References
        ----------
        Singh, A. P., & Gordon, G. J. (2008). Relational learning via collective matrix factorization.
        Proceeding of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
        KDD 08, 650. https://doi.org/10.1145/1401890.1401969

        Wang, Y., Yanchunzhangvueduau, E., & Zhou, B. (2017).
        Semi-supervised collective matrix factorization for topic detection and document clustering.
        """
    def __init__(self, n_components=None, x_init=None, y_init=None, solver='mu', alpha='auto',
                 beta_loss='frobenius', tol=1e-4, max_iter=600,
                 random_state=None, l1_reg=0., l2_reg=0., verbose=0,
                 U_non_negative=True, V_non_negative=True, Z_non_negative=True,
                 x_link="linear", y_link="linear", hessian_pertubation=0.2, sg_sample_ratio=1.):
        self.n_components = n_components
        self.x_init = x_init
        self.y_init = y_init
        self.solver = solver
        self.alpha = alpha
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.verbose = verbose
        self.U_non_negative = U_non_negative
        self.V_non_negative = V_non_negative
        self.Z_non_negative = Z_non_negative
        self.x_link = x_link
        self.y_link = y_link
        self.hessian_pertubation = hessian_pertubation
        self.sg_sample_ratio = sg_sample_ratio

    def fit_transform(self, X, Y, U=None, V=None, Z=None):
        """Learn a CMF model for the data X and Y and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            First data matrix to be decomposed

        Y : {array-like, sparse matrix}, shape (n_features, n_labels)
            Second data matrix to be decomposed

        U : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        V : array-like, shape (n_features, n_components)
            If init='custom', it is used as initial guess for the solution.

        Z : array-like, shape (n_labels, n_components)
            If init='custom', it is used as initial guess for the solution.

        Returns
        -------
        U : array, shape (n_samples, n_components)
            Transformed data.

        V : array, shape (n_features, n_components)
            Transformed data.

        Z : array, shape (n_labels, n_components)
            Transformed data.

        """
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
        Y = check_array(Y, accept_sparse=('csr', 'csc'), dtype=float)

        if X.shape[1] != Y.shape[0]:
            raise ValueError("Expected X.shape[1] == Y.shape[0], " +
                             "found X.shape = {}, Y.shape = {}".format(X.shape, Y.shape))

        U, V, Z, n_iter_ = collective_matrix_factorization(
            X=X, Y=Y, U=U, V=V, Z=Z, n_components=self.n_components,
            x_init=self.x_init, y_init=self.y_init,
            solver=self.solver, alpha=self.alpha, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, l1_reg=self.l1_reg,
            l2_reg=self.l2_reg, random_state=self.random_state, verbose=self.verbose,
            U_non_negative=self.U_non_negative, V_non_negative=self.V_non_negative,
            Z_non_negative=self.Z_non_negative,
            x_link=self.x_link, y_link=self.y_link,
            hessian_pertubation=self.hessian_pertubation, sg_sample_ratio=self.sg_sample_ratio)

        self.reconstruction_err_ = compute_factorization_error(X, U, V.T, self.x_link, self.beta_loss)
        self.reconstruction_err_ += compute_factorization_error(Y, V, Z.T, self.y_link, self.beta_loss)

        self.n_components_ = U.shape[1]
        self.x_weights = U
        self.components = V
        self.y_weights = Z
        self.n_iter_ = n_iter_

        return U, V, Z

    def fit(self, X, Y, **params):
        """Learn a CMF model for the data X and Y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        Y : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        self.fit_transform(X, Y, **params)
        return self

    def transform(self, X, Y):
        """Fit on X/Y while keeping components matrix (V) constant.
        If only fitting on either X or Y, set the other to None.
        """
        assert(hasattr(self, "components"))
        update_U = X is not None
        update_Z = Y is not None
        alpha = 1 if Y is None else 0 if X is None else "auto"
        U = None if update_U else self.x_weights
        Z = None if update_Z else self.y_weights
        U, V, Z, n_iter_ = collective_matrix_factorization(
            X=X, Y=Y, U=U, V=self.components, Z=Z,
            n_components=self.n_components, x_init="custom", y_init="custom",
            solver=self.solver, alpha=alpha, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, l1_reg=self.l1_reg,
            l2_reg=self.l2_reg, random_state=self.random_state, verbose=self.verbose,
            U_non_negative=self.U_non_negative, V_non_negative=self.V_non_negative,
            Z_non_negative=self.Z_non_negative,
            update_U=update_U, update_V=False, update_Z=update_Z,
            x_link=self.x_link, y_link=self.y_link,
            hessian_pertubation=self.hessian_pertubation, sg_sample_ratio=self.sg_sample_ratio)
        return U, V, Z

    def print_topic_terms(self, vectorizer, topn_words=10, importances=True):
        """For interpreting the results when using CMF for labeled topic modeling.
        Prints out the topics acquired along with the words included.

        Parameters
        ----------
        vectorizer : {sklearn.VectorizerMixin}
            The vectorizer that maps words to tfidf/count vectors.
            CMF currently expects the input to be preprocessed using CountVectorizer
            or TfidfVectorizer and will use the vectorizer's mapping to map
            word idxs back to the original words

        topn_words : int, default: 10
            Number of words to display per topic
            (words are chosen in order of weight within topic)

        importances : bool, default: True
            Whether to print the importances along with the topics.
        """
        idx_to_word = np.array(vectorizer.get_feature_names())
        if importances:
            _print_topic_terms_with_importances_from_matrices(
                self.x_weights, self.y_weights,
                idx_to_word, topn_words=topn_words)
        else:
            _print_topic_terms_from_matrix(
                self.x_weights, idx_to_word,
                topn_words=topn_words)


if __name__ == '__main__':
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(20, 15))
    Y = np.abs(rng.randn(15, 10))
    model = CMF(n_components=10, solver='newton', x_link="logit",
                random_state=42)
    U, V, Z = model.fit_transform(X, Y)
