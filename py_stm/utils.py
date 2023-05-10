import numpy as np
import patsy as pt
import pandas as pd
from patsy import ModelDesc

from scipy.sparse import csc_matrix, issparse
from scipy.special import polygamma, psi

from .stm import logger

def get_max_id(documents):
    """Get the highest feature id that appears in the corpus.
    Parameters
    ----------
    corpus : iterable of iterable of (int, numeric)
    Collection of texts in BoW format.
    Returns
    -------
    int
            Highest feature id.
    Notes
    -----
    For empty `corpus` return -1.
    """
    maxid = -1
    for document in documents:
        if document:
            maxid = max(maxid, max(fieldid for fieldid, _ in document))
    return maxid

def is_valid_kappa_prior(kappa_prior):
    """Check to see if kappa_prior arg is valid
    A valid value is either 'L1' or 'Jeffreys'
    Parameters
    ----------
    kappa_prior : str
            kappa prior type for content covariance
    Returns
    -------
    bool
            True if valid False if invalid
    """
    return kappa_prior in ["L1", "Jeffreys"]


def is_valid_gamma_prior(gamma_prior):
    """Check to see if gamma_prior arg is valid
    A valid value is either 'L1' or 'Pooled'
    Parameters
    ----------
    gamma_prior : str
            gamma prior type for relevance
    Returns
    -------
    bool
            True if valid False if invalid
    """
    return gamma_prior in ["Pooled", "L1"]


def is_valid_init_type(init_type):
    """Check to see if init_type arg is valid
    A valid value is either 'Spectral', 'LDA', 'Random', or 'Custom'
    Parameters
    ----------
    init_type : str
            initialization type for topic model
    Returns
    -------
    bool
            True if valid False if invalid
    """
    return init_type in ["Spectral", "LDA", "Random", "Custom"]


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

def update_dir_prior(prior, N, logphat, rho):
    """Update a given prior using Newton's method, described in
    `J. Huang: "Maximum Likelihood Estimation of Dirichlet Distribution Parameters"
    <http://jonathan-huang.org/research/dirichlet/dirichlet.pdf>`_.

    Parameters
    ----------
    prior : list of float
        The prior for each possible outcome at the previous iteration (to be updated).
    N : int
        Number of observations.
    logphat : list of float
        Log probabilities for the current estimation, also called "observed sufficient statistics".
    rho : float
        Learning rate.

    Returns
    -------
    list of float
        The updated prior.

    """
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    updated_prior = rho * dprior + prior
    if all(updated_prior > 0):
        prior = updated_prior
    else:
        logger.warning("updated prior is not positive")
    return prior