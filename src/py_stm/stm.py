import logging

from . import utils

logger = logging.getLogger(__name__)

class Stm:
	def __init__(self, documents=None, vocab=None, K=100,
        	prevalence=None, content=None, data=None,
                init_type=["Spectral", "LDA", "Random", "Custom"], seed=None,
                max_em_its=500, emtol=1e-5,
                verbose=True, reportevery=5,
                LDAbeta=True, interactions=True,
                ngroups=1, model=None,
                gamma_prior=["Pooled", "L1"], sigma_prior=0,
                kappa_prior=["L1", "Jeffreys"], control=[]):
		"""
		Parameters
		----------
		documents : iterable, optional
			List of documents to be processed. Each document can be a string or a list of words.
		vocab : list, optional
			List of words in the vocabulary. If not provided, the vocabulary will be extracted from the documents.
		K : int, optional
			Number of topics to be extracted from the documents.
		prevalence : array-like, optional
			Prior prevalence of topics. If not provided, uniform prior will be used.
		content : array-like, optional
			Content of documents. If not provided, content will be extracted from the documents.
		data : array-like, optional
			Additional data associated with documents. Can be used for covariate information.
		init_type : list of str, optional
			Initialization type for topic model. Can be one of ["Spectral", "LDA", "Random", "Custom"].
		seed : int, optional
			Random seed for reproducibility.
		max_em_its : int, optional
			Maximum number of EM iterations.
		emtol : float, optional
			Convergence tolerance for EM.
		verbose : bool, optional
			Verbosity level.
		reportevery : int, optional
			Number of iterations between progress reports.
		LDAbeta : bool, optional
			Use LDA-style beta update.
		interactions : bool, optional
			Use interaction term updates.
		ngroups : int, optional
			Number of groups for structured topic model.
		model : STM, optional
			Pre-trained STM model for custom initialization.
		gamma_prior : list of str, optional
			Gamma prior type for prevalence. Can be one of ["Pooled", "L1"].
		sigma_prior : float, optional
			Sigma prior for content covariance.
		kappa_prior : list of str, optional
			Kappa prior type for content covariance. Can be one of ["L1", "Jeffreys"].
		control : list, optional
			Control list for advanced options.
		"""

		self.vocab = vocab
		if documents is None and self.vocab is None:
			raise ValueError('at least one of documents/vocab must be specified to establish input space dimensionality')
		
		if self.vocab is None:
			logger.warning('no vocab word id mapping provided; initializing from documents')
			self.vocab = utils.vocab_from_documents(documents)
			self.num_terms = len(self.vocab)
		elif len(self.vocab) > 0:
			self.num_terms = 1 + max(self.vocab.keys())
		else:
			self.num_terms = 0
		
		if self.num_terms == 0:
			raise ValueError('cannot compute STM over an empty collection (no terms)')
