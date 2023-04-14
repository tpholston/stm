import numpy as np
from scipy.sparse import csc_matrix, issparse


class FakeDict:
    """Objects of this class act as dictionaries that map integer->str(integer), for a specified
    range of integers <0, num_terms). This is meant to avoid allocating real dictionaries when
    `num_terms` is huge, which is a waste of memory.
    """

    def __init__(self, num_terms):
        """
        Parameters
        ----------
        num_terms : int
                Number of terms.
        """
        self.num_terms = num_terms

    def __str__(self):
        return f"{self.__class__.__name__}<num_terms={self.num_terms}>"

    def __getitem__(self, val):
        if 0 <= val < self.num_terms:
            return str(val)
        raise ValueError(
            f"internal id out of bounds ({val}, expected <0..{self.num}))")

    def __contains__(self, val):
        return 0 <= val < self.num_terms

    def iteritems(self):
        """Iterate over all keys and values.
        Yields
        ------
        (int, str)
                Pair of (id, token).
        """
        for i in range(self.num_terms):
            yield i, str(i)

    def keys(self):
        """Override the `dict.keys()`, which is used to determine the maximum internal id of a corpus,
        i.e. the vocabulary dimensionality.
        Returns
        -------
        list of int
                Highest id, packed in list.
        Notes
        -----
        To avoid materializing the whole `range(0, self.num_terms)`,
        this returns the highest id = `[self.num_terms - 1]` only.
        """
        return [self.num_terms - 1]

    def __len__(self):
        return self.num_terms

    def get(self, val, default=None):
        """Get the string form of the id in the dictionary. Returns None if not in dictionary
        Parameters
        ----------
        val : int
                key id in dictionary.
        Returns
        -------
        str
                String of dictionary id.
        Notes
        -----
        Will return None if the id is not in the dictionary (if is not in the range 0 to num_terms).
        """
        if 0 <= val < self.num_terms:
            return str(val)
        return default


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


def vocab_from_documents(documents):
    """Scan documents for all word ids that appear in it, then construct a mapping
    which maps each `word_id` -> `str(word_id)`.
    Parameters
    ----------
    documents : iterable of iterable of (int, numeric)
            Collection of texts in BoW format.
    Returns
    ------
    vocab : :class:`~gensim.utils.FakeDict`
            "Fake" mapping which maps each `word_id` -> `str(word_id)`.
    Warnings
    --------
    This function is used whenever *words* need to be displayed (as opposed to just their ids)
    but no `word_id` -> `word` mapping was provided. The resulting mapping only covers words actually
    used in the corpus, up to the highest `word_id` found.
    """
    num_terms = 1 + get_max_id(documents)
    vocab = FakeDict(num_terms)
    return vocab


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


def make_top_matrix(x, data=None):
    """
    Create a sparse matrix from a prevalence-covariate design matrix in Python.

    Parameters:
    x (str or array-like): Either a formula or a matrix.
    data (pandas.DataFrame, optional): The data to use with the formula. Default is None.

    Returns:
    scipy.sparse.csc_matrix: A compressed sparse column matrix.

    Raises:
    TypeError: If the input is not a formula or a matrix.

    Notes:
    - If x is a formula, the function first checks if it is a prevalence formula and then creates a sparse model matrix using the Matrix package.
    - If this fails, the function creates a standard model matrix using the stats package.
    - If the resulting matrix is less than 50% sparse or has fewer than 50 columns, the function converts it to a standard matrix.
    - If x is a matrix, the function checks if it has an intercept in the first column.
    - If it does, the function returns the matrix as a sparse matrix.
    - Otherwise, the function adds a column of 1s to the matrix and returns it as a sparse matrix.
    """
    if isinstance(x, str):
        termobj = data[x]
        if termobj.response == 1:
            raise ValueError(
                "Response variables should not be included in prevalence formula.")
        xmat = termobj.sparse_model_matrix()
        propSparse = 1 - np.count_nonzero(xmat) / xmat.size
        if propSparse < 0.5 or xmat.shape[1] < 50:
            xmat = xmat.toarray()
        return csc_matrix(xmat)
    elif isinstance(x, np.ndarray):
        if np.all(x[:, 0] == 1):
            return csc_matrix(x)
        else:
            xmat = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
            return csc_matrix(xmat)
    else:
        raise TypeError("Input must be a formula or a matrix.")


def to_internal_stm_format(documents, vocab, data):
    """Convert parameters to a common format for the STM:

    Parameters
    ----------
    documents : sparse matrix or numpy array
            Input documents data, can be either a sparse matrix or a numpy array.
    vocab : dict or None
            Vocabulary dictionary or None, if None and documents is a numpy array, a vocabulary will be created.
    data : any
            Additional data for STM. Can be used for covariate information.

    Returns
    -------
    tuple
            Converted documents data, converted vocabulary dictionary or created vocabulary for numpy array documents, additional data for STM
    """
    if issparse(documents):
        # for sparse matrix will recreate vocab regardless of if provided
        vocab = list(range(documents.shape[1]))

    if vocab is None and isinstance(documents, np.ndarray):
        # for normal matrix
        unique_words = np.unique(documents)
        vocab = {word: i for i, word in enumerate(unique_words)}
    return documents, vocab, data
