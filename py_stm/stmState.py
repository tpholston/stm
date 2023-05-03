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
    def __init__(self, eta, shape, dtype=np.float32):
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
        self.eta = eta.astype(dtype, copy=False)
        self.sstats = np.zeros(shape, dtype=dtype)
        self.numdocs = 0
        self.dtype = dtype

    def reset(self):
        """Prepare the state for a new EM iteration (reset sufficient stats)."""
        self.sstats[:] = 0.0
        self.numdocs = 0

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
        self.sstats += other.sstats
        self.numdocs += other.numdocs

    def blend(self, rhot, other, targetsize=None):
        """Merge the current state with another one using a weighted average for the sufficient statistics.

        The number of documents is stretched in both state objects, so that they are of comparable magnitude.
        This procedure corresponds to the stochastic gradient update from
        `'Online Learning for LDA' by Hoffman et al.`_, see equations (5) and (9).

        Parameters
        ----------
        rhot : float
            Weight of the `other` state in the computed average. A value of 0.0 means that `other`
            is completely ignored. A value of 1.0 means `self` is completely ignored.
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.
        targetsize : int, optional
            The number of documents to stretch both states to.

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        # stretch the incoming n*phi counts to target size
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            logger.info("merging changes from %i documents into a model of %i documents", other.numdocs, targetsize)
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats

        self.numdocs = targetsize

    def blend2(self, rhot, other, targetsize=None):
        """Merge the current state with another one using a weighted sum for the sufficient statistics.

        In contrast to :meth:`~gensim.models.ldamodel.LdaState.blend`, the sufficient statistics are not scaled
        prior to aggregation.

        Parameters
        ----------
        rhot : float
            Unused.
        other : :class:`~gensim.models.ldamodel.LdaState`
            The state object with which the current one will be merged.
        targetsize : int, optional
            The number of documents to stretch both states to.

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # merge the two matrices by summing
        self.sstats += other.sstats
        self.numdocs = targetsize

    def get_lambda(self):
        """Get the parameters of the posterior over the topics, also referred to as "the topics".

        Returns
        -------
        numpy.ndarray
            Parameters of the posterior probability over topics.

        """
        return self.eta + self.sstats

    def get_Elogbeta(self):
        """Get the log (posterior) probabilities for each topic.

        Returns
        -------
        numpy.ndarray
            Posterior probabilities for each topic.
        """
        return dirichlet_expectation(self.get_lambda())

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