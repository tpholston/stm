import time
import numpy as np
import patsy as pt
import pandas as pd
import scipy as sp

from patsy import ModelDesc
from qpsolvers import solve_qp

from scipy.sparse import csc_matrix, issparse, diags, csr_array, csr_matrix
from sklearn.preprocessing import normalize
from scipy.special import polygamma, psi, logsumexp
from scipy import optimize


from .stm import logger

def make_top_matrix(prevalence, data=None):
    """
    Create a matrix (or sparse matrix) from a prevalence-covariate design matrix in Python.

    Parameters:
    prevalence (str or array-like): 
        Either a formula or a matrix.
    data (pandas.DataFrame, optional): 
        The data to use with the formula. Default is None.

    Returns:
    scipy.sparse.csc_matrix or np.ndarray: 
        A covariate matrix.

    Raises:
    TypeError: 
        If the input is not a formula or a matrix.
    """
    if isinstance(prevalence, str):
        # TODO: This component needs heavy testing
        desc = ModelDesc.from_formula(prevalence)
        if desc.lhs_termlist != []:
            raise ValueError("response variables should not be included in prevalence formula.")
        prevalence_mat = pt.dmatrix(prevalence, data=data, return_type='matrix')
        prop_sparse = 1 - np.count_nonzero(prevalence_mat) / prevalence_mat.size
        if prop_sparse < 0.5 or prevalence_mat.shape[1] < 50:
            return prevalence_mat
        return csc_matrix(prevalence_mat)
    if isinstance(prevalence, np.ndarray):
        # Check if it has an intercept in the first column
        if np.allclose(prevalence[:, 0], np.ones(prevalence.shape[0])):
            return prevalence
        else:
            return np.hstack((np.ones((prevalence.shape[0], 1)), prevalence))
    elif isinstance(prevalence, pd.DataFrame):
        # Check if it has an intercept column
        if np.allclose(prevalence.iloc[:, 0], np.ones(prevalence.shape[0])):
            return prevalence.to_numpy()
        else:
            return np.hstack((np.ones((prevalence.shape[0], 1)), prevalence.to_numpy()))
    raise TypeError("input must be a formula (str), dataframe, or a matrix.")


def to_internal_stm_format(documents, vocab, data):
    """Convert parameters to a common format for the STM:

    Parameters
    ----------
    documents : sparse matrix or numpy array
        Input documents data, can be either a sparse matrix or a numpy array.
    vocab : dict or None
        Vocabulary dictionary or None, if None and documents 
        is a numpy array, a vocabulary will be created.
    data : any
        Additional data for STM. Can be used for covariate information.

    Returns
    -------
    tuple
        Converted documents data, converted vocabulary dictionary or 
        created vocabulary for numpy array documents, additional data for STM
    """
    if issparse(documents):
        # for sparse matrix will recreate vocab regardless of if provided
        vocab = list(range(documents.shape[1]))

    if vocab is None and isinstance(documents, np.ndarray):
        # for normal matrix
        unique_words = np.unique(documents)
        vocab = {word: i for i, word in enumerate(unique_words)}
    return documents, vocab, data

# TODO: similar still
def compute_gram_matrix(doc_word_freq):
    """
    Compute the gram matrix for a given document-term frequency matrix.

    Parameters
    ----------
    doc_word_matrix : scipy.sparse.csr_matrix
        A sparse matrix representing a document-term frequency matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        The computed gram matrix.
    """
    word_counts = doc_word_freq.sum(axis=1)
    divisor = np.array(word_counts) * np.array(word_counts - 1)

    # Convert to sparse matrices to save time
    doc_word_freq = csr_matrix(doc_word_freq)
    Htilde = csr_matrix(doc_word_freq / np.sqrt(divisor))
    Hhat = diags(np.array(np.sum(doc_word_freq / divisor, axis=0)).flatten(), 0)

    # Compute Q matrix (may take some time)
    Q = Htilde.T @ Htilde - Hhat

    # Normalize Q
    assert np.all(Q.sum(axis=1) > 0), "Encountered zeroes in Q row sums, cannot normalize."
    normalize(Q, copy=False)

    return Q

def fast_anchor(Qbar, num_topics):
    """
    Find anchor words from a gram matrix.

    Given a gram matrix Q, this function returns anchor words for the specified number of topics.
    The function selects the anchor words in a deterministic manner without using randomness.
    It does not support resuming the process from the middle.

    Parameters:
    -----------
    Qbar : csr_matrix
        The gram matrix represented as a sparse CSR matrix.
    num_topics : int
        The number of anchor words to select.

    Returns:
    --------
    numpy.ndarray:
        An array of indices representing the anchor words (rows of Q).
    """
    anchors = np.zeros(num_topics, dtype=np.int64)
    row_squared_sum = Qbar.power(2).sum(axis=0)

    for i in range(num_topics):
        anchors[i] = np.argmax(row_squared_sum)
        max_val = row_squared_sum.max()
        normalizer = 1 / np.sqrt(max_val)

        Qbar[anchors[i]] = Qbar[anchors[i]] * normalizer
        inner_products = np.dot(Qbar, Qbar[anchors[i], :].T)
        project = np.dot(inner_products, Qbar[anchors[i], :])

        project[anchors, :] = 0
        Qbar = np.subtract(Qbar, project)
        row_squared_sum = Qbar.power(2).sum(axis=0)
        row_squared_sum[:, anchors] = 0

    return anchors

def recover_l2(Qbar, anchors, word_probabilities):
    """
    Recover L2-normalized word-topic distributions.

    Given a gram matrix Qbat, anchor words, and word probabilities,
    this function computes the L2-normalized word-topic distributions.

    Parameters:
    -----------
    Qbar : csr_matrix
        The gram matrix represented as a sparse CSR matrix.
    anchors : numpy.ndarray 
        Anchor vector of indices for rows of Qbar containing anchors.
        An array of indices representing the anchor words.
    word_probabilities : numpy.ndarray
        An array of probabilities for each word.

    Returns:
    --------
    numpy.ndarray:
        The L2-normalized word-topic distributions.

    """
    M = Qbar[np.int64(anchors)]
    P = np.dot(M, M.T).toarray()

    G = np.eye(M.shape[0])
    h = np.zeros(M.shape[0])
    
    condprob = np.empty(Qbar.shape[0], dtype=np.ndarray)
    for i in range(Qbar.shape[0]):
        if i in anchors:
            vec = np.repeat(0, P.shape[0])
            vec[np.where(anchors == i)] = 1
            condprob[i] = vec
        else:
            y = Qbar[i]
            q = (M @ y.T).toarray().flatten()
            solution = solve_qp(P=P, q=q, G=G, h=h, solver='quadprog')
            condprob[i] = -1 * solution

    weights = np.vstack(condprob)
    A = weights.T * word_probabilities
    A = A.T / np.sum(A, axis=1)

    assert np.any(A > 0), "Negative probabilities for some words."
    assert np.any(A < 1), "Word probabilities larger than one."

    return A.T


def optimize_lambda(num_topics, lambda_, mu, word_count, beta_doc, sigma_inv):
    """
    Optimize the variational parameter eta given the likelihood and the gradient function.

    Parameters:
    ----------
    num_topics : int
        integer representing the number of topics in the model
    lambda_ : np.ndarray
        Mean topic distribution of the document d.
    mu : np.ndarray
        Prior mean parameter.
    word_count : np.ndarray
        Count of words of the document d.
    beta_doc : np.ndarray
        Word-topic distribution for the document d.
    sigma_inv : np.array
        Inverse of sigma

    Returns:
    -------
    OptimizeResult:
        Result of the optimization.
    """
    def objective(lambda_, num_topics, word_count, mu, beta_doc, sigma_inv):
        """
        Objective function for the variational update q(eta).

        Parameters:
        ----------
        num_topics : int
            integer representing the number of topics in the model.
        lambda_ : np.ndarray
            Mean topic distribution of the document d.
        word_count : np.ndarray
            Count of words of the document d.
        mu : np.ndarray
            Prior mean parameter.
        beta_doc : np.ndarray
            Word-topic distribution for the document d.
        sigma_inv : np.array
            Inverse of sigma.

        Returns:
        -------
        float:
            Function value for the objective.
        """
        lambda_extended = np.insert(lambda_, num_topics - 1, 0)
        N_doc = int(np.sum(word_count))

        return np.float64(
            (0.5 * (lambda_extended[:-1] - mu).T @ sigma_inv @ (lambda_extended[:-1] - mu))
            - (
                np.dot(
                    word_count,
                    lambda_extended.max() + np.log(np.exp(lambda_extended - lambda_extended.max()) @ beta_doc),
                )
                - N_doc * logsumexp(lambda_extended)
            )
        )

    def gradient(lambda_, num_topics, word_count, mu, beta_doc, sigma_inv):
        """
        Gradient for the objective of the variational update q(eta).

        Parameters:
        ----------
        num_topics : int
            integer representing the number of topics in the model.
        lambda_ : np.ndarray
            Mean topic distribution of the document d.
        word_count : np.ndarray
            Count of words of the document d.
        mu : np.ndarray
            Prior mean parameter.
        beta_doc : np.ndarray
            Word-topic distribution for the document d.
        sigma_inv : np.array
            Inverse of sigma.

        Returns:
        -------
        np.ndarray:
            Gradient of the objective.
        """
        lambda_extended = np.insert(lambda_, num_topics - 1, 0)
        return np.array(
            np.float64(
                sigma_inv @ (lambda_extended[:-1] - mu)
                - (
                    beta_doc @ (word_count / np.sum(beta_doc.T, axis=1))
                    - (np.sum(word_count) / np.sum(np.exp(lambda_extended))) * np.exp(lambda_extended)
                )[:-1]
            )
        )

    return optimize.minimize(
        objective, x0=lambda_, args=(num_topics, word_count, mu, beta_doc, sigma_inv), jac=gradient, method="BFGS"
    )

def optimize_nu(L):
    """
    Optimize the variance-covariance matrix for the variational distribution q(eta|lambda, nu).

    Parameters:
    ----------
    L : np.array
        Lower triangular matrix of the Cholesky decomposition.

    Returns:
    -------
    np.array: 
        Variance-covariance matrix for the variational distribution.
    """
    L_transpose = L.T
    nu = np.linalg.inv(np.triu(L_transpose))
    nu = np.dot(nu, nu.T)
    return nu

def update_z(num_topics, lambda_, topic_word_distribution, word_count):
    """
    Compute the update for the variational latent parameter z.

    Paramters
    ---------
    num_topics : int
        integer representing the number of topics in the model.
    lambda_ : np.array
        1D-array representing prior to the document-topic distribution.
    topic_word_distribution : np.array 
        2D-array (K by V) containing the topic-word distribution for a specific document.
    word_count : np.array 
        1D-array representing the word counts in the document.

    Returns
    -------
    phi : np.array
        Update for the variational latent parameter z
    """
    lambda_extended = np.insert(lambda_, num_topics - 1, 0)
    a = np.multiply(topic_word_distribution.T, np.exp(lambda_extended)).T
    b = np.multiply(a, (np.sqrt(word_count) / np.sum(a, axis=0)))
    phi = np.multiply(b, np.sqrt(word_count).T)
    assert np.all(phi >= 0), "Some values of phi are zero or NaN."
    return phi

def lower_bound(num_topics, L, mu, word_count, topic_word_distribution, lambda_, sigma_inv, sigma_entropy):
    """
    Compute the evidence lower bound (ELBO) and update for the variational latent parameter z.

    Parameters:
    ----------
    num_topics : int
        integer representing the number of topics in the model.
    L : np.array
        Lower triangular matrix of the Cholesky decomposition.
    mu : np.array
        Mean parameter for the logistic normal distribution.
    word_count : np.array
        Word counts for each document.
    topic_word_distribution : np.array
        Topic-word distribution for a specific document.
    lambda_ : np.array
        Prior to the document-topic distribution.
    sigma_inv : np.array
        Inverse of sigma.
    sigma_entropy : np.array
        Entropy of sigma.

    Returns:
    -------
    bound : float
        Evidence lower bound (ELBO).
    phi : np.array
        Update for the variational latent parameter z.
    """
    lambda_extended = np.insert(lambda_, num_topics - 1, 0)
    theta = stable_softmax(lambda_extended)

    detTerm = -np.sum(np.log(L.diagonal()))
    diff = lambda_ - mu

    beta_temp_kv = topic_word_distribution * np.exp(lambda_extended)[:, None]
    bound = (np.log(theta[None:,] @ beta_temp_kv) @ word_count + detTerm - 0.5 * diff.T @ sigma_inv @ diff - sigma_entropy)
    return bound

def decompose_hessian(hessian):
    """
    Decompose the Hessian matrix using Cholesky decomposition.

    Parameters:
    ----------
    hessian : np.array
        The Hessian matrix to be decomposed.

    Returns:
    -------
    np.array:
        The lower triangular matrix of the Cholesky decomposition.
    """

    try:
        L = np.linalg.cholesky(hessian)
    except:
        try:
            L = np.linalg.cholesky(make_positive_definite(hessian))
        except:
            L = sp.linalg.cholesky(make_positive_definite(hessian) + 1e-5 * np.eye(hessian.shape[0]))
    return L

def make_positive_definite(matrix):
    """
    Convert a matrix to be positive definite.

    The following are necessary (but not sufficient) conditions for a Hermitian matrix A
    (which by definition has real diagonal elements a_(ii)) to be positive definite.

    1. a_(ii) > 0 for all i,
    2. a_(ii) + a_(jj) > 2 * abs(Re[a_(ij)]) for i != j,
    3. The element with the largest modulus lies on the main diagonal,
    4. det(A) > 0.

    Returns:
    -------
    np.array:
        The positive definite matrix.

    Raises:
    ------
    ValueError:
        If the matrix is not positive definite.
    """
    diagonal_vec = matrix.diagonal()
    magnitudes = np.sum(abs(matrix), 1) - abs(diagonal_vec)
    diagonal_vec = np.where(diagonal_vec < magnitudes, magnitudes, diagonal_vec)
    np.fill_diagonal(matrix, diagonal_vec)
    return matrix

def hessian(num_topics, lambda_, word_count, topic_word_distribution, sigma_inv):
    """
    Compute the Hessian matrix for the objective function.

    Parameters:
    ----------
    num_topics : int
        integer representing the number of topics in the model.
    lambda_ : np.ndarray
        Document-specific prior on topical prevalence of shape (1 x K).
    word_count : np.ndarray
        Document-specific vector of word counts of shape (1 x V_doc).
    topic_word_distribution : np.ndarray
        Document-specific word-topic distribution of shape (K x V_doc).
    sigma_inv : np.array
        Inverse of sigma.

    Returns:
    -------
    np.ndarray:
        Negative Hessian matrix as specified in Roberts et al. (2016b).
    """
    lambda_extended = np.insert(lambda_, num_topics - 1, 0)
    theta = stable_softmax(lambda_extended)

    a = np.transpose(np.multiply(np.transpose(topic_word_distribution), np.exp(lambda_extended)))  # KxV
    b = np.multiply(a, np.transpose(np.sqrt(word_count))) / np.sum(a, 0)  # KxV
    c = np.multiply(b, np.transpose(np.sqrt(word_count)))  # KxV

    hess = b @ b.T - np.sum(word_count) * np.multiply(theta[:, None], theta[None, :])
    np.fill_diagonal(hess, np.diag(hess) - np.sum(c, axis=1) + np.sum(word_count) * theta)

    d = hess[:-1, :-1]
    f = d + sigma_inv

    if not np.all(np.linalg.eigvals(f) > 0):
        f = make_positive_definite(f)
        if not np.all(np.linalg.eigvals(f) > 0):
            np.fill_diagonal(f, np.diag(f) + 1e-5)

    return f


def stable_softmax(x):
    """
    Compute the stable softmax values for each set of scores in x.

    Parameters:
    ----------
    x : np.ndarray
        Input scores.

    Returns:
    -------
    np.ndarray:
        Softmax values.
    """
    x_shifted = x - np.max(x)
    exps = np.exp(x_shifted)
    return exps / np.sum(exps)
