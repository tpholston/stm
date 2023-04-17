import logging
import numpy as np
import patsy as pt
import pandas as pd
from scipy.sparse import issparse

from convergence import Convergence
from covariates import Covariates
from gamma import Gamma
from sigma import Sigma
from kappa import Kappa
from tau import Tau
from init import Init
from . import utils
from constants import LEGAL_ARGS

logger = logging.getLogger(__name__)


class Stm:
    """
    Class representing a topic model using the Structural Topic Model (STM) algorithm.

    ...

    Methods
    -------
    __init__(self, documents, vocab=None, data=None, num_topics=10, alpha=0.01, eta=0.01, diagnostics=True, vocabulary=None)
            Initialize the STM model with given parameters
    estimate(self, max_em_iter=100, em_tol=1e-04, initialize=True, verbose=False)
            Estimate the parameters of the STM model using the Expectation-Maximization (EM) algorithm
    infer_topics(self, new_documents)
            Infer topic proportions for new documents using the trained STM model
    get_topic_proportions(self)
            Get the estimated topic proportions for the input documents
    get_topic_assignments(self)
            Get the estimated topic assignments for the input documents
    ...
    """

    def __init__(self, documents=None, vocab=None, k=100,
                 prevalence=None, content=None, data=None,
                 init_type="LDA", seed=None,
                 max_em_its=500, emtol=1e-5,
                 verbose=True, reportevery=5,
                 lda_beta=True, interactions=True,
                 ngroups=1, model=None,
                 gamma_prior="Pooled", sigma_prior=0,
                 kappa_prior="L1", control={}):
        """
        Parameters
        ----------
        documents : iterable, optional
                List of documents to be processed. Each document is a list of words. Can be a sparse matrix.
        vocab : list, optional
                List of words in the vocabulary. If not provided, the vocabulary
                will be extracted from the documents. Not necessary for sparse matrix.
        k : int, optional
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

        if not utils.is_valid_gamma_prior(gamma_prior):
            raise ValueError('invalid gamma prior. Should be one of ["L1", "Pooled"]')
		
        if not utils.is_valid_init_type(init_type):
            raise ValueError('invalid init type. Should be one of ["Spectral", "LDA", "Random", "Custom"]')
		
        if not utils.is_valid_kappa_prior(kappa_prior):
            raise ValueError('invalid kappa prior. Should be one of ["L1", "Jeffreys"]')

        # Documents
        if documents is None:
            raise ValueError("documents must be specified.")
        if isinstance(documents, list):
            documents = np.array(documents)
        if not (isinstance(documents, np.ndarray) or issparse(documents)):
            raise ValueError(
                "documents must be of type list, np.ndarray, or sparse matrix.")
        if isinstance(documents, np.ndarray):
            if not all(isinstance(doc, np.ndarray)
                       and doc.ndim == 2 for doc in documents):
                raise ValueError(
                    "each list element in documents must be a matrix. See documentation."
                )
            if any(np.any(np.diff(doc[0, :]) == 0) for doc in documents):
                raise ValueError(
                    "duplicate term indices within a document. See documentation for proper format."
                )
        n = len(self.documents)

        # Convert to a standard internal STM format here
        self.documents, self.vocab, self.data = utils.to_internal_stm_format(
            documents, vocab, data)

        # Extract and check the word indices
        wcountvec = np.concatenate(
            [np.repeat(x[0, :], x[1, :]) for x in self.documents])
        if not np.issubdtype(
                wcountvec.dtype,
                np.integer) or np.any(
                wcountvec <= 0):
            raise ValueError("word indices are not positive integers.")
        if not np.array_equal(
            np.unique(wcountvec), np.arange(
                1, len(
                np.unique(wcountvec)) + 1)):
            raise ValueError(
                "word indices must be sequential integers starting with 1.")
        v, wcounts = np.unique(wcountvec, return_counts=True)
        v = len(v)
        if len(vocab) != v:
            raise ValueError(
                "vocab length does not match observed word indices.")

        # Check the number of topics
        if k is None:
            raise ValueError("k, the number of topics, is required.")
        if k != 0:
            if not isinstance(k, int) or k <= 1:
                raise ValueError(
                    "k must be a positive integer greater than 1."
                )
            if k == 2:
                logger.warning(
                    "k=2 is equivalent to a unidimensional scaling model which you may prefer."
                )
        else:
            if init_type != "Spectral":
                raise ValueError(
                    "Topic selection method can only be used with init_type='Spectral'."
                )

        # Iterations
        if not (isinstance(max_em_its, int) and max_em_its >= 0):
            raise ValueError(
                "Max EM iterations must be a single non-negative integer."
            )

        # Verbose
        if not isinstance(verbose, bool):
            raise ValueError("Verbose must be a boolean value.")

        # Now we parse both sets of covariates
        if prevalence is not None:
            if not isinstance(
                    prevalence,
                    np.ndarray) and not isinstance(
                    prevalence,
                    str):
                raise ValueError(
                    "prevalence covariates must be specified as a model matrix or as a formula."
                )
            prevalence_mat = utils.make_top_matrix(prevalence, data)
            if np.isnan(np.count_nonzero(prevalence_mat)):
                raise ValueError("missing values in prevalence covariates.")
        else:
            prevalence_mat = None

        if content is not None:
            if isinstance(content, str):
                if len(content.split("~")) == 2:
                    raise ValueError("response variables should not be included in prevalence formula.")
                termobj = pt.dmatrix(content, data=data)
                if len(termobj.design_info.terms) != 1:
                    raise ValueError("currently content can only contain one variable.")
                if data is None:
                    yvar = pd.Series(termobj[:, 0], name=termobj.design_info.column_name(0))
                else:
                    char = termobj.design_info.column_name(0)
                    yvar = data[char]
                yvar = pd.factorize(yvar)[0]
            else:
                yvar = pd.factorize(content)[0]
            if np.any(pd.isna(yvar)):
                raise ValueError(
                    "your content covariate contains missing values. All values of " +
                    "the content covariate must be observed."
                )
            yvarlevels = np.unique(yvar)
            betaindex = yvar
        else:
            yvarlevels = None
            betaindex = np.repeat(1, len(documents))
        a = len(np.unique(betaindex))  # define the number of aspects

        # check for dimension agreement
        ny = len(betaindex)
        nx = self.num_documents if prevalence_mat is None else prevalence_mat.shape[0]
        if self.num_documents != nx or self.num_documents != ny:
            raise ValueError(
                f"number of observations in content covariate ({ny}) prevalence covariate ({nx}) " + 
                "and documents ({self.num_documents}) are not all equal."
            )
        
        # some additional sanity checks
        if not isinstance(lda_beta, bool):
            raise ValueError("lda_beta must be boolean")
        if not isinstance(interactions, bool):
            raise ValueError("interactions variable must be boolean")
        if sigma_prior < 0 or sigma_prior > 1:
            raise ValueError("sigma_prior must be between 0 and 1")
        if model is not None:
            if(max_em_its <= model.max_em_its):
                raise ValueError(
                    "when restarting a model, max_em_its represents the total iterations of the model " +
                    "and thus must be greater than the length of the original run."
                )
            
		# set a lot of instance variables

        # dim ? these need better variable names just 
        # trying to follow the STM code right now though
        self.k = int(k) # K -> number of topics
        self.a = a # A -> number of aspects
        self.v = v # V -> number of vocab
        self.n = n # N -> number of documents
        self.wcounts = wcounts

        self.verbose = verbose
        self.reportevery = reportevery

        # convergance related variables
        self.convergence = Convergence(max_em_its, emtol, allow_negative_change=True)

        # covariates related variables
        self.covariates = Covariates(prevalence_mat, betaindex, yvarlevels, prevalence)

        # gamma related variables
        self.gamma = Gamma(gamma_prior)

        # sigma related variables
        self.sigma = Sigma(sigma_prior)

        # kappa related variables
        self.kappa = Kappa(lda_beta, interactions)

        # tau related variables
        self.tau = Tau(kappa_prior)

        # init realted variables
        self.init = Init(init_type, 50/k)

        self.seed = seed
        self.ngroups = ngroups

        if self.init.mode == "Spectral" and self.v > 10000:
            self.init.max_v = 10000

        if self.gamma.mode == "L1" and prevalence_mat.shape[1] <= 2:
            raise ValueError("cannot use L1 penalization in prevalence model with 2 or fewer covariates.")
        
        # is there a covariate on top?
        if prevalence is None:
            self.gamma.mode = "CTM" # without covariates has to be estimating the mean.
        
        # is there a covariate on the bottom?
        if content is None:
            self.kappa.interactions = False # can't have interactions without a covariate
        else:
            self.kappa.lda_beta = False # can't do LDA topics with a covariate
        
        for key, value in control.items():
            if key not in LEGAL_ARGS:
                raise ValueError(f"Argument {key} not matched")
            if key == "tau_maxit":
                self.tau.maxit = value
            if key == "tau_tol":
                self.tau.tol = value
            if key == "fixed_intercept":
                self.kappa.fixed_intercept = value
            if key == "kappa_enet":
                self.tau.enet = value
            if key == "kappa_mstep_maxit":
                self.kappa.mstep_maxit = value
            if key == "kappa_mstep_tol":
                self.kappa.mstep_tol = value
            if key == "nlambda":
                self.tau.nlambda = value
            if key == "lambda_min_ratio":
                self.tau.lambda_min_ratio = value
            if key == "ic_k":
                self.tau.ic_k = value
            if key == "gamma_enet":
                self.gamma.enet = value
            if key == "gamma_ic_k":
                self.gamma.ic_k = value
            if key == "nits":
                self.init.nits = value
            if key == "burnin":
                self.init.burnin = value
            if key == "alpha":
                self.init.alpha = value
            if key == "eta":
                self.init.eta = value
            if key == "contrast":
                self.kappa.contrast = value
            if key == "rp_s":
                self.init.s = value
            if key == "rp_p":
                self.init.p = value
            if key == "rp_d_group_size":
                self.init.d_group_size = value
            if key == "spectral_IRP" and value:
                self.init.mode = "SpectralRP"
            if key == "recoverEG" and not value:
                self.init.recoverEG = value
            if key == "max_v" and value:
                self.init.max_v = value
                if self.init.max_v > self.v:
                    raise ValueError("max_v cannot be larger than the vocabulary")
            if key == "tSNE_init_dims" and value:
                self.init.tSNE_init_dims = value
            if key == "tSNE_perplexity" and value:
                self.init.tSNE_perplexity = value
            if key == "gamma_maxits":
                self.gamma.max_its = value
            if key == "allow_negative_change":
                self.convergence.allow_negative_change = value
            if key == "custom_beta":
                if self.init.mode != "Custom":
                    logger.warning("custom beta supplied, setting init argument to Custom.")
                    self.init.mode = "Custom"
                self.init.custom = value                

		# set the random seed if passed in as a parameter
        if seed is not None:
            np.random.seed(seed)
