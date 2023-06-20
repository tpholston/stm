import logging
import numpy as np
from gensim import utils
from gensim.matutils import dirichlet_expectation
from .stm import logger

class StmState(utils.SaveLoad):
    """Encapsulates the state of a single STM object. This is designed
    to be used for distributed computation. 
    
    May be sent over network so keep lightweight.
    """
    def __init__(self, beta, sigma, mu, theta_shape, lambda_shape, 
                 beta_shape, sigma_shape, dtype=np.float32):
        """
        Parameters
        ----------
        eta : numpy.ndarray
            The prior probabilities assigned to each term.
        shape : tuple of (int, int)
            Shape of the sufficient statistics: (number of topics to be found, number of terms in the vocabulary).
        dtype : type
            Overrides the numpy array default types.
        """
        self.beta = beta.astype(dtype, copy=False)
        self.sigma = sigma.astype(dtype, copy=False)
        self.mu = mu.astype(dtype, copy=False)
        self.theta = np.zeros(theta_shape, dtype=dtype)
        self.lambda_ = np.zeros(lambda_shape, dtype=dtype)
        self.beta_ss = np.zeros(beta_shape, dtype=dtype)
        self.sigma_ss = np.zeros(sigma_shape, dtype=dtype)
        self.numdocs = 0
        self.calculated_bounds = 0
        self.dtype = dtype

    def reset(self):
        """Prepare the state for a new EM iteration (reset sufficient stats)."""
        self.theta[:] = 0.0
        self.lambda_[:] = 0.0
        self.beta_ss[:] = 0.0
        self.sigma_ss[:] = 0.0
        self.numdocs = 0
        self.calculated_bounds = 0

    def merge(self, other):
        """Merge the result of an E step from one node with that of another node (summing up sufficient statistics).

        The merging is trivial and after merging all cluster nodes, we have the
        exact same result as if the computation was run on a single node (no
        approximation).

        Parameters
        ----------
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.

        """
        assert other is not None
        self.theta += other.theta
        self.lambda_ += other.lambda_
        self.beta_ss += other.beta_ss
        self.sigma_ss += other.sigma_ss
        self.calculated_bounds += other.calculated_bounds
        self.numdocs += other.numdocs
    
    def get_lambda(self):
        """TODO: MAKE THIS FUNC DOC
        TODO: ONLY FOR PYLDAVIS COMPATABILITY"""
        return self.beta + self.beta_ss


    @classmethod
    def load(cls, fname, *args, **kwargs):
        """Load a previously stored state from disk.

        Overrides :class:`~gensim.utils.SaveLoad.load` by enforcing the `dtype` parameter
        to ensure backwards compatibility.

        Parameters
        ----------
        fname : str
            Path to file that contains the needed object.
        args : object
            Positional parameters to be propagated to class:`~gensim.utils.SaveLoad.load`
        kwargs : object
            Key-word parameters to be propagated to class:`~gensim.utils.SaveLoad.load`

        Returns
        -------
        :class:`~gensim.models.ldamodel.LdaState`
            The state loaded from the given file.

        """
        result = super(StmState, cls).load(fname, *args, **kwargs)

        # dtype could be absent in old models
        if not hasattr(result, 'dtype'):
            result.dtype = np.float64  # float64 was implicitly used before (because it's the default in numpy)
            logging.info("dtype was not set in saved %s file %s, assuming np.float64", result.__class__.__name__, fname)

        return result