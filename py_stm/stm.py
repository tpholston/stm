import logging
import time
import os
import numpy as np
from patsy import ModelDesc, dmatrix
import pandas as pd
from collections import defaultdict
from operator import itemgetter

from scipy.sparse import csr_matrix
from scipy.special import logsumexp
from scipy.linalg import cholesky
from scipy.stats import rankdata

from gensim.models import basemodel, CoherenceModel
from gensim import interfaces
from gensim.matutils import (
    argsort, kullback_leibler, jensen_shannon, jaccard_distance, hellinger
)
import gensim.utils as gensim_utils
from gensim.models.callbacks import Callback

from sklearn.linear_model import Lasso, Ridge, LinearRegression, PoissonRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)

from .stmState import StmState
from . import utils

class StmModel(interfaces.TransformationABC, basemodel.BaseTopicModel):
    def __init__(self, corpus=None, num_topics=100, id2word=None, 
                 metadata=None, prevalence=None, content=None,
                 chunksize=2000, passes=1, update_every=0,
                 decay=0.5, offset=1.0, eval_every=10, minimum_probability=0.01,
                 random_state=None, minimum_phi_value=0.01, per_word_topics=False, 
                 callbacks=None, dtype=np.float32, init_mode="Spectral", 
                 max_vocab=5000, interactions=True, convergence_threshold=1e-5, 
                 LDAbeta=True, sigma_prior=0, model="STM", gamma_prior="L1"):
        """

        Parameters
        ----------
        corpus : iterable of list of (int, float), optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
            If you have a CSC in-memory matrix, you can convert it to a
            streamed corpus with the help of gensim.matutils.Sparse2Corpus.
            If not given, the model is left untrained (presumably because you want to call
            :meth:`~stm.StmModel.update` manually).
        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        id2word : {dict of (int, str), :class:`gensim.corpora.dictionary.Dictionary`}
            Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for
            debugging and topic printing.
        metadata : (pandas.DataFrame, numpy.ndarray), optional
            Optional dataframe containing the prevalence and/or content covariates. 
        prevalence : (pandas.DataFrame, numpy.ndarray, str), optional
            A formula object with no response variable or a matrix containing topic prevalence covariates.
        content : str, optional
            A formula containing a single variable, a factor variable or something which can be coerced
            to a factor indicating the category of the content variable for each document.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        passes : int, optional
            Number of passes through the corpus during training.
        update_every : int, optional
            Number of documents to be iterated through for each update.
            Set to 0 for batch learning, > 1 for online iterative learning.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined.
            Corresponds to :math:`\\kappa` from `'Online Learning for LDA' by Hoffman et al.`_
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        minimum_probability : float, optional
            Topics with a probability lower than this threshold will be filtered out.
        random_state : {np.random.RandomState, int}, optional
            Either a randomState object or a seed to generate one. Useful for reproducibility.
        minimum_phi_value : float, optional
            if `per_word_topics` is True, this represents a lower bound on the term probabilities.
        per_word_topics : bool
            If True, the model also computes a list of topics, sorted in descending order of most likely topics for
            each word, along with their phi values multiplied by the feature length (i.e. word count).
        callbacks : list of :class:`~gensim.models.callbacks.Callback`
            Metric callbacks to log and visualize evaluation metrics of the model during training.
        dtype : {numpy.float16, numpy.float32, numpy.float64}, optional
            Data-type to use during calculations inside model. All inputs are also converted.
        init_mode: TODO:
        """

        self.dtype = np.finfo(dtype).dtype

        # store user-supplied parameters
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = gensim_utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0

        if self.num_terms == 0:
            raise ValueError("cannot compute STM over an empty collection (no terms)")

        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0

        # TODO: these are added. Get rid of unused instance vars
        # TODO: interactions comment
        # TODO: LDAbeta comment
        # TODO: sigma_prior comment
        # TODO: content comment
        # TODO: prevalence comment
        # TODO: max_vocab comment
        # TODO: init_mode comment
        # TODO: convergence_threshold comment
        # TODO: model comment
        # TODO: gamma_prior comment
        self.interactions = interactions
        self.init_mode = init_mode
        self.max_vocab = max_vocab
        self.LDAbeta = LDAbeta
        self.last_bounds = []
        self.content = content
        self.prevalence = prevalence
        self.sigma_prior = sigma_prior
        self.convergence_threshold = convergence_threshold
        self.model = model
        self.mode = gamma_prior

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.callbacks = callbacks

        # covariates TODO: PLEASE TEST
        if self.prevalence is not None:
            self.covariates = self.prevalence
        else:
            self.model = "CTM"
            self.covariates = None

        if self.content is not None:
            self.LDAbeta = False
            if isinstance(content, pd.Series): #TODO: ensure this is 1d array
                yvar = content.astype("category")
                self.yvarlevels = set(yvar)
                self.betaindex = np.array(yvar.cat.codes)
            else:
                raise ValueError("Content needs to be type pd.DataFrame. Currently, content can only be one column.")
            
            if yvar.isnull().any().any():
                raise ValueError("Your content covariate contains missing values. All values of the content covariate must be observed.")
        else:
            self.yvarlevels = None
            self.betaindex = np.ones(len(corpus)) # TODO: this needs to be moved. 
            self.interactions = False

        self.random_state = gensim_utils.get_random_state(random_state)

        logger.info("using serial STM version on this node")

        # Initialize the variational distribution q(beta|lambda)
        # TODO: modify this component depending on what we figure out in the inference()
        # self.state.sstats[...] = self.random_state.gamma(100., 1. / 100., (self.num_topics, self.num_terms))
        # self.expElogbeta = np.exp(dirichlet_expectation(self.state.sstats))

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            try:
                self.lencorpus = len(corpus)
            except Exception:
                logger.warning("input corpus stream has no len(); counting documents")
                self.lencorpus = sum(1 for _ in corpus)
            if self.lencorpus == 0:
                logger.warning("StmModel called with an empty corpus")
                return

            start = time.time()

            # hyperparams
            self.init_priors(corpus, self.init_mode, self.max_vocab)
            self.state = StmState(self.beta, (self.num_topics, self.num_terms), dtype=self.dtype)

            self.update(corpus)
            self.add_lifecycle_event(
                "created", msg=f"trained {self} in {time.time() - start:.2f}s",
            )

    def init_priors(self, corpus, init_mode, max_vocab):
        """
        Initialize priors for the Structural Topic Model.

        Depending on the initialization mode specified, this method initializes the necessary priors
        such as beta, mu, sigma, lambda, and theta.

        Parameters
        ----------
        corpus : iterable of list of (int, float)
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        init_mode : str
            The initialization mode to use. Possible values: 'LDA', 'Random', or 'Spectral'.
        max_vocab : int
            Maximum number of top words to select based on probabilities for spectral initialization.

        Returns
        -------
        None
            The method initializes the priors of the Structural Topic Model object.
        """
        if init_mode is None:
            init_mode = 'Spectral'

        if init_mode == 'LDA':
            init_beta = self.lda_init()
            # TODO: init other params for LDA mode
        elif init_mode == 'Random':
            init_beta = self.random_init()
            # TODO: init other params for Random mode 
        elif init_mode == 'Spectral':
            doc_id, word_id, word_freq = [], [], []
            for i, doc in enumerate(corpus):
                for word, freq in doc:
                    doc_id.append(i)
                    word_id.append(word)
                    word_freq.append(freq)

            doc_word_freq = csr_matrix((word_freq, (doc_id, word_id)))
            self.wcounts = np.array(doc_word_freq.sum(axis=0)).flatten()

            self.beta = self.spectral_init(doc_word_freq, max_vocab)
            self.mu = np.zeros((self.lencorpus, self.num_topics - 1))
            self.sigma = np.zeros(((self.num_topics - 1), (self.num_topics - 1)))
            np.fill_diagonal(self.sigma, 20)
            self.lambda_ = np.zeros((self.lencorpus, self.num_topics - 1))
            self.theta = np.zeros((self.lencorpus, self.num_topics))
            
    def spectral_init(self, doc_word_freq, max_vocab):
        """
        Perform spectral initialization.

        Given a document-term frequency matrix, this method computes the gram matrix, selects anchor words,
        and recovers the L2 matrix. It then constructs beta values for each aspect based on the selected top words.

        Parameters
        ----------
        doc_word_freq : numpy.ndarray
            Document-term frequency matrix.
        max_vocab : int
            Maximum number of top words to select based on probabilities.

        Returns
        -------
        numpy.ndarray
            Beta values representing the topic-word distributions for each aspect.
        """
        # Compute word probabilities
        word_probabilities = np.sum(doc_word_freq, axis=0)
        word_probabilities = word_probabilities / np.sum(word_probabilities)
        word_probabilities = np.array(word_probabilities).flatten()

        # Select top words based on probabilities
        top_words_indices = np.argsort(-1 * word_probabilities)[:max_vocab]
        doc_word_freq = doc_word_freq[:, top_words_indices]
        word_probabilities = word_probabilities[top_words_indices]

        # Prepare the Gram matrix
        gram_matrix = utils.compute_gram_matrix(doc_word_freq)

        # Compute anchor words
        anchor_words = utils.fast_anchor(gram_matrix, self.num_topics)

        # Recover L2
        recovered_beta = utils.recover_l2(gram_matrix, anchor_words, word_probabilities)

        if top_words_indices is not None:
            updated_beta = np.zeros(self.num_topics * len(self.id2word)).reshape(self.num_topics, len(self.id2word))
            updated_beta[:, top_words_indices] = recovered_beta
            updated_beta += 0.001 / len(self.id2word)
            recovered_beta = updated_beta / np.sum(updated_beta)

        # Create a list of beta values for each aspect
        if self.interactions:
            recovered_beta = np.array([recovered_beta.copy() for _ in set(self.betaindex)])

        return recovered_beta

    def __str__(self):
        """Get a string representation of the current object.

        Returns
        -------
        str
            Human readable representation of the most important model parameters.

        """
        return "%s<num_terms=%s, num_topics=%s, chunksize=%s>" % (
            self.__class__.__name__, self.num_terms, self.num_topics, self.chunksize
        )

    # TODO: modify function comment. STATE
    def inference(self, chunk, mu, betaindex, lambda_, theta, beta_ss, sigma_ss, collect_sstats=False):
        """
        Given a chunk of sparse document vectors, estimate gamma (parameters controlling the topic weights)
        for each document in the chunk.

        This function does not modify the model. The whole input chunk of documents is assumed to fit in RAM;
        chunking of a large corpus must be done earlier in the pipeline. It avoids computing the `phi` variational
        parameter directly using the optimization presented in
        `Lee, Seung: Algorithms for non-negative matrix factorization"
        <https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf>`_.

        Parameters
        ----------
        chunk : list of list of (int, float)
            The corpus chunk on which the inference step will be performed.
        collect_sstats : bool, optional
            If set to True, also collect (and return) sufficient statistics needed to update the model's topic-word
            distributions.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first element is always returned and it corresponds to the beta_ss matrix. The second element corresponds to
            the sigma_ss matrix and is only returned if `collect_sstats` == True.
        """
        try:
            len(chunk)
        except TypeError:
            # convert iterators/generators to plain list, so we have len() etc.
            chunk = list(chunk)

        if len(chunk) > 1:
            logger.debug("Performing inference on a chunk of %i documents", len(chunk))

        # Precalculate common components
        while True:
            try:
                sigobj = np.linalg.cholesky(self.sigma)
                self.sigma_entropy = np.sum(np.log(np.diag(sigobj)))
                self.sigma_inv = np.linalg.inv(sigobj).T * np.linalg.inv(sigobj)
                break
            except:
                logger.error("Cholesky Decomposition failed because Sigma is not positive definite.")
                self.sigma_entropy = 0.5 * np.linalg.slogdet(self.sigma)[1]  # part 2 of ELBO
                self.sigma_inv = cholesky(self.sigma)  # part 3 of ELBO

        # Initialize sufficient statistics
        calculated_bounds = []

        start_time = time.time()
        # For each document d, update gamma and phi
        integer_types = (int, np.integer,)
        # epsilon = np.finfo(self.dtype).eps
        for d, doc in enumerate(chunk):
            if len(doc) > 0 and not isinstance(doc[0][0], integer_types):
                # Make sure the term IDs are ints; otherwise, np will raise an error
                ids = [int(idx) for idx, _ in doc]
            else:
                ids = [idx for idx, _ in doc]
            cts = np.fromiter((cnt for _, cnt in doc), dtype=self.dtype, count=len(doc))

            aspect = betaindex[d] if self.content is not None else None

            beta_doc_kv = self.get_topic_word_distribution(ids, aspect=aspect)
            
            assert np.all(beta_doc_kv >= 0), "Some entries of beta are negative or NaN."

            res = utils.optimize_lambda(
                num_topics=self.num_topics,
                lambda_=lambda_[d],
                mu=mu[d],
                word_count=cts,
                beta_doc=beta_doc_kv,
                sigma_inv=self.sigma_inv,
            )

            lambda_[d] = res.x
            theta[d] = np.exp(np.insert(res.x, self.num_topics - 1, 0)) / np.sum(
                np.exp(np.insert(res.x, self.num_topics - 1, 0))
            )

            # Compute Hessian, Phi, and Lower Bound
            hess_i = utils.hessian(
                num_topics=self.num_topics,
                lambda_=lambda_[d], 
                word_count=cts, 
                topic_word_distribution=beta_doc_kv,
                sigma_inv=self.sigma_inv,
            )
            L_i = utils.decompose_hessian(hess_i)

            bound_i = utils.lower_bound(
                num_topics=self.num_topics,
                L=L_i,
                mu=mu[d],
                word_count=cts,
                topic_word_distribution=beta_doc_kv,
                lambda_=lambda_[d],
                sigma_inv=self.sigma_inv,
                sigma_entropy=self.sigma_entropy,
            )

            nu = utils.optimize_nu(L_i)

            self.phi = utils.update_z(
                num_topics = self.num_topics,
                lambda_=lambda_[d],
                topic_word_distribution=beta_doc_kv,
                word_count=cts,
            )

            calculated_bounds.append(bound_i)

            # Update sufficient statistics
            sigma_ss += nu
            if self.interactions:
                beta_ss[aspect][:, np.array(np.int64(ids))] += self.phi
            else:
                try:
                    beta_ss[:, np.array(np.int64(ids))] += self.phi
                except RuntimeWarning:
                    breakpoint()

        self.bound = np.sum(calculated_bounds)
        self.last_bounds.append(self.bound)
        elapsed_time = np.round((time.time() - start_time), 3)
        logging.info(f"Lower Bound: {self.bound}")
        logging.info(f"Completed E-Step in {elapsed_time} seconds.\n")

        return beta_ss, sigma_ss, lambda_, theta

    # TODO: modify function comment. STATE
    def do_estep(self, chunk, mu, betaindex, lambda_, theta, beta_ss, sigma_ss, state=None):
        """Perform inference on a chunk of documents, and accumulate the collected sufficient statistics.

        Parameters
        ----------
        chunk : list of list of (int, float)
            The corpus chunk on which the inference step will be performed.
        state : :class:`~stm.StmState`, optional
            The state to be updated with the newly accumulated sufficient statistics. If none, the models
            `self.state` is updated.
        TODO:

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first element is always returned and it corresponds to the beta_ss matrix. The second element corresponds to
            the sigma_ss matrix and is only returned if `collect_sstats` == True.
        """
        if state is None:
            state = self.state
        beta_ss, sigma_ss, lambda_, theta = self.inference(chunk, mu, betaindex, lambda_, theta, beta_ss, sigma_ss, collect_sstats=True)
        # TODO: here we should be updating state and updating sstats
        #state.sstats += sstats
        #state.numdocs += gamma.shape[0]  # avoids calling len(chunk) on a generator
        #assert gamma.dtype == self.dtype
        return beta_ss, sigma_ss, lambda_, theta

    def get_topic_word_distribution(self, words, aspect):
        """
        Get the topic-word distribution for a document with the respective topical content covariate (aspect).

        Parameters:
        ----------
        words : np.ndarray
            1D-array with word indices for a specific document.
        aspect : int or float
            Topical content covariate for a specific document.

        Returns:
        -------
        np.ndarray:
            Topic-word distribution for a specific document, based on word indices and aspect.
        """
        if self.interactions:
            topic_word_distribution = self.beta[aspect][:, np.array(np.int64(words))]
        else:
            # TODO: I think the issue is here. Check beta len. Might need to be chunked
            topic_word_distribution = self.beta[:, np.array(np.int64(words))]

        return topic_word_distribution

    # TODO: remove? Idk if this even works for STM
    def log_perplexity(self, chunk, total_docs=None):
        """Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus.

        Also output the calculated statistics, including the perplexity=2^(-bound), to log at INFO level.

        Parameters
        ----------
        chunk : list of list of (int, float)
            The corpus chunk on which the inference step will be performed.
        total_docs : int, optional
            Number of docs used for evaluation of the perplexity.

        Returns
        -------
        numpy.ndarray
            The variational bound score calculated for each word.

        """
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        logger.info(
            "%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words",
            perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words
        )
        return perwordbound

    def update(self, corpus, chunksize=None, decay=None, offset=None,
               passes=None, update_every=None, eval_every=None, convergence_threshold=None):
        """Train the model with new documents, by EM-iterating over the corpus until the topics converge, or until
        the maximum number of allowed iterations is reached. `corpus` must be an iterable.

        In distributed mode, the E step is distributed over a cluster of machines.

        Notes
        -----
        This update also supports updating an already trained model (`self`) with new documents from `corpus`;
        the two models are then merged in proportion to the number of old vs. new documents.
        This feature is still experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of `'Online Learning for LDA' by Hoffman et al.`_
        and is guaranteed to converge for any `decay` in (0.5, 1].
        Additionally, for smaller corpus sizes,
        an increasing `offset` may be beneficial (see Table 1 in the same paper).

        Parameters
        ----------
        corpus : iterable of list of (int, float), optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to update the
            model.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined. Corresponds to :math:`\\kappa` from
            `'Online Learning for LDA' by Hoffman et al.`_
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_
        passes : int, optional
            Number of passes through the corpus during training.
        update_every : int, optional
            Number of documents to be iterated through for each update.
            Set to 0 for batch learning, > 1 for online iterative learning.
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        """
        # use parameters given in constructor, unless user explicitly overrode them
        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every
        if eval_every is None:
            eval_every = self.eval_every
        if convergence_threshold is None:
            convergence_treshold = self.convergence_threshold

        try:
            self.lencorpus = len(corpus)
        except Exception:
            logger.warning("input corpus stream has no len(); counting documents")
            self.lencorpus = sum(1 for _ in corpus)
        if self.lencorpus == 0:
            logger.warning("StmModel.update() called with an empty corpus")
            return
        
        # if corpus wan't provided in class constructor
        if self.beta is None:
            self.beta, self.mu, self.sigma, self.lambda_, self.theta = self.init_priors(self.corpus, self.init_mode, self.max_vocab)
            self.state = StmState(self.beta, (self.num_topics, self.num_terms), dtype=self.dtype)
            
        # Checks for Dimension agreement
        ny = len(self.betaindex)
        nx = self.covariates.shape[0] if self.covariates is not None else self.lencorpus
        if( self.lencorpus != nx or self.lencorpus != ny):
            raise ValueError(f"Number of observations in content covariate ({ny}) prevalence covariate ({nx}) and documents ({self.lencorpus}) are not all equal.")

        if chunksize is None:
            chunksize = min(self.lencorpus, self.chunksize)

        self.state.numdocs += self.lencorpus

        if update_every:
            updatetype = "online"
            if passes == 1:
                updatetype += " (single-pass)"
            else:
                updatetype += " (multi-pass)"
            updateafter = min(self.lencorpus, update_every * chunksize)
        else:
            updatetype = "batch"
            updateafter = self.lencorpus
        evalafter = min(self.lencorpus, (eval_every or 0) * chunksize)

        updates_per_pass = max(1, self.lencorpus / updateafter)
        logger.info(
            "running %s STM training, %s topics, %i passes over "
            "the supplied corpus of %i documents, updating model once "
            "every %i documents, evaluating perplexity every %i documents, "
            "with a convergence threshold of %f",
            updatetype, self.num_topics, passes, self.lencorpus,
            updateafter, evalafter, convergence_threshold
        )

        if updates_per_pass * passes < 10:
            logger.warning(
                "too few updates, training might not converge; "
                "consider increasing the number of passes or iterations to improve accuracy"
            )

        if self.callbacks:
            # pass the list of input callbacks to Callback class
            callback = Callback(self.callbacks)
            callback.set_model(self)
            # initialize metrics list to store metric values after every epoch
            self.metrics = defaultdict(list)

        converged = False
        first_start_time = time.time()
        for pass_ in range(passes):
            other = StmState(self.beta, self.state.sstats.shape, self.dtype)
            dirty = False

            reallen = 0
            chunks = gensim_utils.grouper(corpus, chunksize, dtype=self.dtype)

            sigma_ss = np.zeros(shape=self.sigma.shape)
            beta_ss = np.zeros(shape=self.beta.shape)

            for chunk_no, chunk in enumerate(chunks):
                reallen += len(chunk)  # keep track of how many documents we've processed so far

                # TODO: logging perplexity is probably an important option here
                #if eval_every and ((reallen == lencorpus) or ((chunk_no + 1) % (eval_every) == 0)):
                #    self.log_perplexity(chunk, total_docs=lencorpus)

                logger.info(
                    "PROGRESS: pass %i, at document #%i/%i",
                    pass_, chunk_no * chunksize + len(chunk), self.lencorpus
                )

                # get the document indices for this chunk
                # TODO: go back to make sure this min() works effectively. Should only be relevant for dirty doc chunks.
                chunk_indices = range(chunk_no * chunksize, min(chunk_no * chunksize + len(chunk), self.lencorpus))
                
                chunk_mu = self.mu[chunk_indices]
                chunk_betaindex = self.betaindex[chunk_indices]
                chunk_lambda = self.lambda_[chunk_indices]
                chunk_theta = self.theta[chunk_indices]

                # print(chunk_mu.shape, self.mu.shape)
                # print(chunk_lambda.shape, self.lambda_.shape, chunksize, chunk_no, len(chunk))
                # print(chunk_betaindex.shape, self.betaindex.shape)

                beta_ss, sigma_ss, chunk_lambda, chunk_theta = self.do_estep(chunk, mu=chunk_mu, betaindex=chunk_betaindex, 
                                                                             lambda_=chunk_lambda, theta=chunk_theta, beta_ss=beta_ss, sigma_ss=sigma_ss, state=other)

                # self.mu[chunk_indices] = chunk_mu
                # self.betaindex[chunk_indices] = chunk_betaindex
                self.lambda_[chunk_indices] = chunk_lambda
                self.theta[chunk_indices] = chunk_theta

                dirty = True
                del chunk

                # perform an M step. determine when based on update_every, don't do this after every chunk
                if update_every and (chunk_no + 1) % (update_every) == 0:
                    self.do_mstep(other, beta_ss, sigma_ss, pass_ > 0)
                    del other  # frees up memory
                    other = StmState(self.beta, self.state.sstats.shape, self.dtype)
                    dirty = False
                    # dirty_doc_indices = doc_indices  # store the current indices in case dirty

            if self.is_converged(pass_):
                converged = True
                self.time_processed = time.time() - first_start_time
                logging.info(
                    f"model converged in iteration {pass_} after {self.time_processed}s"
                )
                break            

            if reallen != self.lencorpus:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")

            # append current epoch's metric values
            if self.callbacks:
                current_metrics = callback.on_epoch_end(pass_)
                for metric, value in current_metrics.items():
                    self.metrics[metric].append(value)

            if dirty:
                # get the covariates for the "dirty" documents
                # dirty_doc_covariates = self.covariates[dirty_doc_indices]
                # dirty_word_covariates = self.yvarlevels[dirty_doc_indices]
                # finish any remaining updates
                self.do_mstep(other, beta_ss, sigma_ss, pass_ > 0)
                del other
                dirty = False

        if not converged:
            self.time_processed = time.time() - first_start_time
            logging.info(
                f"maximum number of passes ({passes}) reached after {self.time_processed} seconds"
            )

    # TODO: modify
    def do_mstep(self, other, beta_ss, sigma_ss, extra_pass=False):
        """Maximization step: use linear interpolation between the existing topics and
        collected sufficient statistics in `other` to update the topics.

        Parameters
        ----------
        other : :class:`~stm.StmModel`
            The model whose sufficient statistics will be used to update the topics.
        beta_ss : np.ndarray
            beta sufficicent statistics
        sigma_ss : np.ndarray
            sigma sufficient statistics
        extra_pass : bool, optional
            Whether this step required an additional pass over the corpus.

        """
        logger.debug("updating topics")

        if not extra_pass:
            # only update if this isn't an additional pass
            self.num_updates += other.numdocs

        start_time = time.time()

        self.update_mu()

        self.update_sigma(nu=sigma_ss)

        self.update_beta(beta_ss)

        logging.info(f"Completed M-Step in {np.round((time.time() - start_time), 3)} seconds. \n")

    def update_mu(self, intercept=True):
        """
        Update the mean parameter for the document-specific logistic normal distribution.

        Parameters:
        ----------
        intercept : bool, optional
            Whether or not an intercept is included in the model.

        Raises:
        -------
        ValueError:
            If the model is not "CTM", "STM", or if the mode is not specified correctly.
        """
        if self.model == "CTM":
            # Use the mean for all documents
            self.mu = np.repeat(np.mean(self.lambda_, axis=0)[None, :], self.lencorpus, axis=0)
            return

        try:
            self.covariates = self.covariates.astype("category")
        except:
            pass

        prev_cov = np.array(self.covariates)[:, None]  # Prepare 1D array for one-hot encoding (OHE) by making it 2D

        # Remove empty dimension
        if len(prev_cov.shape) > 2:
            prev_cov = np.squeeze(prev_cov, axis=1)

        if not np.array_equal(prev_cov, prev_cov.astype(bool)):
            enc = OneHotEncoder(handle_unknown="ignore")  # Create OHE
            prev_cov = enc.fit_transform(prev_cov).toarray()  # Fit OHE

        if self.mode == "L1":
            linear_model = Lasso(
                alpha=1, fit_intercept=intercept
            )
            fitted_model = linear_model.fit(prev_cov, self.lambda_)

        elif self.mode == "L2":
            linear_model = Ridge(
                alpha=0.1, fit_intercept=intercept
            )
            fitted_model = linear_model.fit(prev_cov, self.lambda_)

        elif self.mode == "OLS":
            linear_model = LinearRegression(
                fit_intercept=intercept
            )
            fitted_model = linear_model.fit(prev_cov, self.lambda_)
        else:
            raise ValueError('Updating the topical prevalence parameter requires a mode. Choose from "L1", "L2", or "OLS".')

        # Adjust design matrix if intercept is estimated
        if intercept:
            self.gamma = np.column_stack((fitted_model.intercept_, fitted_model.coef_))
            design_matrix = np.c_[np.ones(prev_cov.shape[0]), prev_cov]

        self.gamma = fitted_model.coef_
        design_matrix = prev_cov

        self.mu = design_matrix @ self.gamma.T

    def update_sigma(self, nu):
        """
        Update the variance-covariance matrix for the logistic normal distribution of topical prevalence.

        Parameters:
        ----------
        nu : _type_
            Variance-covariance for the variational document-topic distribution.

        Raises:
        -------
        AssertionError:
            If the weight is not defined between 0 and 1.
        """
        assert 0 <= self.sigma_prior <= 1, 'Weight needs to be defined between 0 and 1.'

        covariance = (self.lambda_ - self.mu).T @ (self.lambda_ - self.mu)
        covariance = np.array(covariance, dtype=np.float64)

        sigma = (covariance + nu) / self.lencorpus
        sigma = np.array(sigma, dtype=np.float64)

        self.sigma = np.diag(np.diag(sigma)) * self.sigma_prior + (1 - self.sigma_prior) * sigma

    def update_beta(self, beta_ss):
        """
        Update the topic-word distribution beta.

        Parameters:
        ----------
        beta_ss : np.ndarray
            Sufficient statistic for word-topic distribution.

        Notes:
        ------
        If self.LDAbeta is True, row-normalization of beta is performed for the update.
        If self.LDAbeta is False, distributed Poisson Regression is used for the updates.
        """
        if self.LDAbeta:
            assert np.any(np.sum(beta_ss, axis=1) >= 0), "Break here"
            row_sums = np.sum(beta_ss, axis=1)[:, None]
            # TODO: check if nan_to_num alters beta. 
            self.beta = np.nan_to_num(self.beta)
            self.beta = np.divide(
                beta_ss, row_sums, out=np.zeros_like(beta_ss), where=row_sums != 0
            )
        else:
            self.distributed_poisson_regression(beta_ss=beta_ss)

    def distributed_poisson_regression(self, beta_ss):
        """
        Perform distributed Poisson regression for updating kappa and beta accordingly.

        Parameters:
        ----------
        beta_ss : np.ndarray
            Estimated word-topic distribution of the current EM iteration with dimensions K x V.

        Notes:
        ------
        - Uses distributed Poisson regression to estimate coefficients for kappa and beta.
        - Supports different cases for topic models and topic-aspect models.
        - Handles fixed intercept and calculates predictions using the estimated coefficients.
        """
        interact = True
        fixed_intercept = True
        alpha = 250  # Corresponds to `lambda` in glmnet
        max_iterations = int(1e4)
        tolerance = 1e-5
        num_aspects = len(self.yvarlevels)  
        counts = csr_matrix(np.vstack(beta_ss))  # Dimensions: (A*K) x V # TODO: Enable dynamic creation of 'counts'

        if num_aspects == 1:  # Topic Model
            covar = np.diag(np.ones(self.num_topics))
        else:  # Topic-Aspect Models
            # if not contrast:
            # Topics
            veci = np.arange(0, counts.shape[0])
            vecj = np.tile(np.arange(0, self.num_topics), num_aspects)
            # Aspects
            veci = np.concatenate((veci, np.arange(0, counts.shape[0])))
            vecj = np.concatenate(
                (
                    vecj,
                    np.repeat(
                        np.arange(self.num_topics, self.num_topics + num_aspects),
                        self.num_topics,
                    ),
                )
            )
            if interact:
                veci = np.concatenate((veci, np.arange(0, counts.shape[0])))
                vecj = np.concatenate(
                    (
                        vecj,
                        np.arange(
                            self.num_topics + num_aspects,
                            self.num_topics + num_aspects + counts.shape[0],
                        ),
                    )
                )  # TODO: Remove +1 at the end, make shapes fit anyway
            vecv = np.ones(len(veci))
            covar = csr_matrix((vecv, (veci, vecj)))

        if fixed_intercept:
            m = self.wcounts
            m = np.log(m) - np.log(np.sum(m))
        else:
            m = 0

        # Distributed Poissons
        # TODO: Scale the data for convergence
        out = []
        # Now iterate over the vocabulary
        for i in range(counts.shape[1]):
            if np.all(m == 0):
                fit_intercept = True
            else:
                fit_intercept = False
            mod = None
            clf = make_pipeline(StandardScaler(with_mean=False), PoissonRegressor(
                fit_intercept=fit_intercept,
                max_iter=np.int64(max_iterations),
                tol=tolerance,
                alpha=np.int64(alpha),
            ))
            mod = clf.fit(covar, counts[:, [i]].A.flatten())
            # if it didn't converge, increase nlambda paths by 20%
            # if(is.null(mod)) nlambda <- nlambda + floor(.2*nlambda)
            # print(f'Estimated coefficients for word {i}.')
            # print(mod.coef_)
            # (0) out.append(mod.params)
            out.append(mod.named_steps['poissonregressor'].coef_)

        # Put all regression results together
        coef = np.stack(out, axis=1)

        # Separate intercept from the coefficients
        if not fixed_intercept:
            m = coef[0]
            coef = coef[1:]

        # Set kappa
        self.kappa = coef

        # Predict
        linpred = covar @ coef
        linpred = m + linpred
        explinpred = np.exp(linpred)
        beta = explinpred / np.sum(explinpred, axis=1)[:, np.newaxis]

        # Retain former structure for beta
        self.beta = np.array(np.split(beta, num_aspects, axis=0))

    
    def is_converged(self, iteration, convergence=None):
        """
        Check if the EM algorithm has converged based on the change in the objective function.

        Parameters:
        ----------
        iteration : int
            Current iteration of the EM algorithm.
        convergence : float, optional
            Threshold for convergence check, defaults to None.

        Returns:
        -------
        bool:
            True if the algorithm has converged, False otherwise.

        Notes:
        ------
        - The convergence check is based on the relative change in the objective function.
        - Requires at least two iterations to perform the check.
        """
        if iteration < 1:
            return False

        new_bound = self.bound
        old_bound = self.last_bounds[-2]

        convergence_check = np.abs((new_bound - old_bound) / np.abs(old_bound))
        logging.info(f"relative change: {convergence_check}")
        if convergence_check < self.convergence_threshold:
            return True
        else:
            return False

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """Get a representation for selected topics.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics to be returned. Unlike LSA, there is no natural ordering between the topics in STM.
            The returned topics subset of all topics is therefore arbitrary and may change between two STM
            training runs.
        num_words : int, optional
            Number of words to be presented for each topic. These will be the most relevant words (assigned the highest
            probability for each topic).
        log : bool, optional
            Whether the output is also logged, besides being returned.
        formatted : bool, optional
            Whether the topic representations should be formatted as strings. If False, they are returned as
            2 tuples of (word, probability).

        Returns
        -------
        list of {str, tuple of (str, float)}
            a list of topics, each represented either as a string (when `formatted` == True) or word-probability
            pairs.

        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)

            # add a little random jitter, to randomize results around the same eta
            sort_eta = self.eta + 0.0001 * self.random_state.rand(len(self.eta))
            # random_state.rand returns float64, but converting back to dtype won't speed up anything

            sorted_topics = list(argsort(sort_eta))
            chosen_topics = sorted_topics[:num_topics // 2] + sorted_topics[-num_topics // 2:]

        shown = []

        topic = self.beta + self.sstats
        for i in chosen_topics:
            topic_ = topic[i]
            topic_ = topic_ / topic_.sum()  # normalize to probability distribution
            bestn = argsort(topic_, num_words, reverse=True)
            topic_ = [(self.id2word[id], topic_[id]) for id in bestn]
            if formatted:
                topic_ = ' + '.join('%.3f*"%s"' % (v, k) for k, v in topic_)

            shown.append((i, topic_))
            if log:
                logger.info("topic #%i (%.3f): %s", i, self.eta[i], topic_)

        return shown

    def show_topic(self, topicid, topn=10):
        """Get the representation for a single topic. Words here are the actual strings, in constrast to
        :meth:`~stm.StmModel.get_topic_terms` that represents words by their vocabulary ID.

        Parameters
        ----------
        topicid : int
            The ID of the topic to be returned
        topn : int, optional
            Number of the most significant words that are associated with the topic.

        Returns
        -------
        list of (str, float)
            Word - probability pairs for the most relevant words generated by the topic.

        """
        return [(self.id2word[id], value) for id, value in self.get_topic_terms(topicid, topn)]

    def get_topics(self):
        """Get the term-topic matrix learned during inference.

        Returns
        -------
        numpy.ndarray
            The probability for each word in each topic, shape (`num_topics`, `vocabulary_size`).

        """
        topics = self.state.get_lambda()
        return topics / topics.sum(axis=1)[:, None]

    def get_topic_terms(self, topicid, topn=10):
        """Get the representation for a single topic. Words the integer IDs, in constrast to
        :meth:`~stm.StmModel.show_topic` that represents words by the actual strings.

        Parameters
        ----------
        topicid : int
            The ID of the topic to be returned
        topn : int, optional
            Number of the most significant words that are associated with the topic.

        Returns
        -------
        list of (int, float)
            Word ID - probability pairs for the most relevant words generated by the topic.

        """
        topic = self.get_topics()[topicid]
        topic = topic / topic.sum()  # normalize to probability distribution
        bestn = argsort(topic, topn, reverse=True)
        return [(idx, topic[idx]) for idx in bestn]

    def top_topics(self, corpus=None, texts=None, dictionary=None, window_size=None,
                   coherence='u_mass', topn=20, processes=-1):
        """Get the topics with the highest coherence score the coherence for each topic.

        Parameters
        ----------
        corpus : iterable of list of (int, float), optional
            Corpus in BoW format.
        texts : list of list of str, optional
            Tokenized texts, needed for coherence models that use sliding window based (i.e. coherence=`c_something`)
            probability estimator .
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Gensim dictionary mapping of id word to create corpus.
            If `model.id2word` is present, this is not needed. If both are provided, passed `dictionary` will be used.
        window_size : int, optional
            Is the size of the window to be used for coherence measures using boolean sliding window as their
            probability estimator. For 'u_mass' this doesn't matter.
            If None - the default window sizes are used which are: 'c_v' - 110, 'c_uci' - 10, 'c_npmi' - 10.
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            Coherence measure to be used.
            Fastest method - 'u_mass', 'c_uci' also known as `c_pmi`.
            For 'u_mass' corpus should be provided, if texts is provided, it will be converted to corpus
            using the dictionary. For 'c_v', 'c_uci' and 'c_npmi' `texts` should be provided (`corpus` isn't needed)
        topn : int, optional
            Integer corresponding to the number of top words to be extracted from each topic.
        processes : int, optional
            Number of processes to use for probability estimation phase, any value less than 1 will be interpreted as
            num_cpus - 1.

        Returns
        -------
        list of (list of (int, str), float)
            Each element in the list is a pair of a topic representation and its coherence score. Topic representations
            are distributions of words, represented as a list of pairs of word IDs and their probabilities.

        """
        cm = CoherenceModel(
            model=self, corpus=corpus, texts=texts, dictionary=dictionary,
            window_size=window_size, coherence=coherence, topn=topn,
            processes=processes
        )
        coherence_scores = cm.get_coherence_per_topic()

        str_topics = []
        for topic in self.get_topics():  # topic = array of vocab_size floats, one per term
            bestn = argsort(topic, topn=topn, reverse=True)  # top terms for topic
            beststr = [(topic[_id], self.id2word[_id]) for _id in bestn]  # membership, token
            str_topics.append(beststr)  # list of topn (float membership, token) tuples

        scored_topics = zip(str_topics, coherence_scores)
        return sorted(scored_topics, key=lambda tup: tup[1], reverse=True)

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        """Get the topic distribution for the given document.

        Parameters
        ----------
        bow : corpus : list of (int, float)
            The document in BOW format.
        minimum_probability : float
            Topics with an assigned probability lower than this threshold will be discarded.
        minimum_phi_value : float
            If `per_word_topics` is True, this represents a lower bound on the term probabilities that are included.
             If set to None, a value of 1e-8 is used to prevent 0s.
        per_word_topics : bool
            If True, this function will also return two extra lists as explained in the "Returns" section.

        Returns
        -------
        list of (int, float)
            Topic distribution for the whole document. Each element in the list is a pair of a topic's id, and
            the probability that was assigned to it.
        list of (int, list of (int, float), optional
            Most probable topics per word. Each element in the list is a pair of a word's id, and a list of
            topics sorted by their relevance to this word. Only returned if `per_word_topics` was set to True.
        list of (int, list of float), optional
            Phi relevance values, multiplied by the feature length, for each word-topic combination.
            Each element in the list is a pair of a word's id and a list of the phi values between this word and
            each topic. Only returned if `per_word_topics` was set to True.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        if minimum_phi_value is None:
            minimum_phi_value = self.minimum_probability
        minimum_phi_value = max(minimum_phi_value, 1e-8)  # never allow zero values in sparse output

        # if the input vector is a corpus, return a transformed corpus
        is_corpus, corpus = gensim_utils.is_corpus(bow)
        if is_corpus:
            kwargs = dict(
                per_word_topics=per_word_topics,
                minimum_probability=minimum_probability,
                minimum_phi_value=minimum_phi_value
            )
            return self._apply(corpus, **kwargs)

        gamma, phis = self.inference([bow], collect_sstats=per_word_topics)
        topic_dist = gamma[0] / sum(gamma[0])  # normalize distribution

        document_topics = [
            (topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
            if topicvalue >= minimum_probability
        ]

        if not per_word_topics:
            return document_topics

        word_topic = []  # contains word and corresponding topic
        word_phi = []  # contains word and phi values
        for word_type, weight in bow:
            phi_values = []  # contains (phi_value, topic) pairing to later be sorted
            phi_topic = []  # contains topic and corresponding phi value to be returned 'raw' to user
            for topic_id in range(0, self.num_topics):
                if phis[topic_id][word_type] >= minimum_phi_value:
                    # appends phi values for each topic for that word
                    # these phi values are scaled by feature length
                    phi_values.append((phis[topic_id][word_type], topic_id))
                    phi_topic.append((topic_id, phis[topic_id][word_type]))

            # list with ({word_id => [(topic_0, phi_value), (topic_1, phi_value) ...]).
            word_phi.append((word_type, phi_topic))
            # sorts the topics based on most likely topic
            # returns a list like ({word_id => [topic_id_most_probable, topic_id_second_most_probable, ...]).
            sorted_phi_values = sorted(phi_values, reverse=True)
            topics_sorted = [x[1] for x in sorted_phi_values]
            word_topic.append((word_type, topics_sorted))

        return document_topics, word_topic, word_phi  # returns 2-tuple

    def get_term_topics(self, word_id, minimum_probability=None):
        """Get the most relevant topics to the given word.

        Parameters
        ----------
        word_id : int
            The word for which the topic distribution will be computed.
        minimum_probability : float, optional
            Topics with an assigned probability below this threshold will be discarded.

        Returns
        -------
        list of (int, float)
            The relevant topics represented as pairs of their ID and their assigned probability, sorted
            by relevance to the given word.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        minimum_probability = max(minimum_probability, 1e-8)  # never allow zero values in sparse output

        # if user enters word instead of id in vocab, change to get id
        if isinstance(word_id, str):
            word_id = self.id2word.doc2bow([word_id])[0][0]

        values = []
        for topic_id in range(0, self.num_topics):
            if self.expElogbeta[topic_id][word_id] >= minimum_probability:
                values.append((topic_id, self.expElogbeta[topic_id][word_id]))

        return values

    def diff(self, other, distance="kullback_leibler", num_words=100,
             n_ann_terms=10, diagonal=False, annotation=True, normed=True):
        """Calculate the difference in topic distributions between two models: `self` and `other`.

        Parameters
        ----------
        other : :class:`~stm.StmModel`
            The model which will be compared against the current object.
        distance : {'kullback_leibler', 'hellinger', 'jaccard', 'jensen_shannon'}
            The distance metric to calculate the difference with.
        num_words : int, optional
            The number of most relevant words used if `distance == 'jaccard'`. Also used for annotating topics.
        n_ann_terms : int, optional
            Max number of words in intersection/symmetric difference between topics. Used for annotation.
        diagonal : bool, optional
            Whether we need the difference between identical topics (the diagonal of the difference matrix).
        annotation : bool, optional
            Whether the intersection or difference of words between two topics should be returned.
        normed : bool, optional
            Whether the matrix should be normalized or not.

        Returns
        -------
        numpy.ndarray
            A difference matrix. Each element corresponds to the difference between the two topics,
            shape (`self.num_topics`, `other.num_topics`)
        numpy.ndarray, optional
            Annotation matrix where for each pair we include the word from the intersection of the two topics,
            and the word from the symmetric difference of the two topics. Only included if `annotation == True`.
            Shape (`self.num_topics`, `other_model.num_topics`, 2).

        """
        distances = {
            "kullback_leibler": kullback_leibler,
            "hellinger": hellinger,
            "jaccard": jaccard_distance,
            "jensen_shannon": jensen_shannon
        }

        if distance not in distances:
            valid_keys = ", ".join("`{}`".format(x) for x in distances.keys())
            raise ValueError("Incorrect distance, valid only {}".format(valid_keys))

        if not isinstance(other, self.__class__):
            raise ValueError("The parameter `other` must be of type `{}`".format(self.__name__))

        distance_func = distances[distance]
        d1, d2 = self.get_topics(), other.get_topics()
        t1_size, t2_size = d1.shape[0], d2.shape[0]
        annotation_terms = None

        fst_topics = [{w for (w, _) in self.show_topic(topic, topn=num_words)} for topic in range(t1_size)]
        snd_topics = [{w for (w, _) in other.show_topic(topic, topn=num_words)} for topic in range(t2_size)]

        if distance == "jaccard":
            d1, d2 = fst_topics, snd_topics

        if diagonal:
            assert t1_size == t2_size, \
                "Both input models should have same no. of topics, " \
                "as the diagonal will only be valid in a square matrix"
            # initialize z and annotation array
            z = np.zeros(t1_size)
            if annotation:
                annotation_terms = np.zeros(t1_size, dtype=list)
        else:
            # initialize z and annotation matrix
            z = np.zeros((t1_size, t2_size))
            if annotation:
                annotation_terms = np.zeros((t1_size, t2_size), dtype=list)

        # iterate over each cell in the initialized z and annotation
        for topic in np.ndindex(z.shape):
            topic1 = topic[0]
            if diagonal:
                topic2 = topic1
            else:
                topic2 = topic[1]

            z[topic] = distance_func(d1[topic1], d2[topic2])
            if annotation:
                pos_tokens = fst_topics[topic1] & snd_topics[topic2]
                neg_tokens = fst_topics[topic1].symmetric_difference(snd_topics[topic2])

                pos_tokens = list(pos_tokens)[:min(len(pos_tokens), n_ann_terms)]
                neg_tokens = list(neg_tokens)[:min(len(neg_tokens), n_ann_terms)]

                annotation_terms[topic] = [pos_tokens, neg_tokens]

        if normed:
            if np.abs(np.max(z)) > 1e-8:
                z /= np.max(z)

        return z, annotation_terms
    
    def __getitem__(self, bow, eps=None):
        """Get the topic distribution for the given document.

        Wraps :meth:`~stm.StmModel.get_document_topics` to support an operator style call.
        Uses the model's current state (set using constructor arguments) to fill in the additional arguments of the
        wrapper method.

        Parameters
        ---------
        bow : list of (int, float)
            The document in BOW format.
        eps : float, optional
            Topics with an assigned probability lower than this threshold will be discarded.

        Returns
        -------
        list of (int, float)
            Topic distribution for the given document. Each topic is represented as a pair of its ID and the probability
            assigned to it.

        """
        return self.get_document_topics(bow, eps, self.minimum_phi_value, self.per_word_topics)

    def save(self, fname, ignore=('state', 'dispatcher'), separately=None, *args, **kwargs):
        """Save the model to a file.

        Large internal arrays may be stored into separate files, with `fname` as prefix.

        Notes
        -----
        If you intend to use models across Python 2/3 versions there are a few things to
        keep in mind:

          1. The pickled Python dictionaries will not work across Python versions
          2. The `save` method does not automatically save all numpy arrays separately, only
             those ones that exceed `sep_limit` set in :meth:`~gensim.utils.SaveLoad.save`. The main
             concern here is the `eta` array if for instance using `eta='auto'`.

        Please refer to the `wiki recipes section
        <https://github.com/RaRe-Technologies/gensim/wiki/
        Recipes-&-FAQ#q9-how-do-i-load-a-model-in-python-3-that-was-trained-and-saved-using-python-2>`_
        for an example on how to work around these issues.

        See Also
        --------
        :meth:`~stm.StmModel.load`
            Load model.

        Parameters
        ----------
        fname : str
            Path to the system file where the model will be persisted.
        ignore : tuple of str, optional
            The named attributes in the tuple will be left out of the pickled model. The reason why
            the internal `state` is ignored by default is that it uses its own serialisation rather than the one
            provided by this method.
        separately : {list of str, None}, optional
            If None -  automatically detect large numpy/scipy.sparse arrays in the object being stored, and store
            them into separate files. This avoids pickle memory errors and allows `mmap`'ing large arrays
            back on load efficiently. If list of str - this attributes will be stored in separate files,
            the automatic check is not performed in this case.
        *args
            Positional arguments propagated to :meth:`~gensim.utils.SaveLoad.save`.
        **kwargs
            Key word arguments propagated to :meth:`~gensim.utils.SaveLoad.save`.

        """
        if self.state is not None:
            self.state.save(gensim_utils.smart_extension(fname, '.state'), *args, **kwargs)
        # Save the dictionary separately if not in 'ignore'.
        if 'id2word' not in ignore:
            gensim_utils.pickle(self.id2word, gensim_utils.smart_extension(fname, '.id2word'))

        # make sure 'state', 'id2word' and 'dispatcher' are ignored from the pickled object, even if
        # someone sets the ignore list themselves
        if ignore is not None and ignore:
            if isinstance(ignore, str):
                ignore = [ignore]
            ignore = [e for e in ignore if e]  # make sure None and '' are not in the list
            ignore = list({'state', 'dispatcher', 'id2word'} | set(ignore))
        else:
            ignore = ['state', 'dispatcher', 'id2word']

        # make sure 'expElogbeta' and 'sstats' are ignored from the pickled object, even if
        # someone sets the separately list themselves.
        separately_explicit = ['expElogbeta', 'sstats']
        # Also add 'eta' and 'beta' to separately list if they are set 'auto' or some
        # array manually.
        if (isinstance(self.eta, str) and self.eta == 'auto') or \
                (isinstance(self.eta, np.ndarray) and len(self.eta.shape) != 1):
            separately_explicit.append('eta')
        if (isinstance(self.beta, str) and self.beta == 'auto') or \
                (isinstance(self.beta, np.ndarray) and len(self.beta.shape) != 1):
            separately_explicit.append('beta')
        # Merge separately_explicit with separately.
        if separately:
            if isinstance(separately, str):
                separately = [separately]
            separately = [e for e in separately if e]  # make sure None and '' are not in the list
            separately = list(set(separately_explicit) | set(separately))
        else:
            separately = separately_explicit
        super(StmModel, self).save(fname, ignore=ignore, separately=separately, *args, **kwargs)

    @classmethod
    def load(cls, fname, *args, **kwargs):
        """Load a previously saved :class:`stm.StmModel` from file.

        See Also
        --------
        :meth:`~stm.StmModel.save`
            Save model.

        Parameters
        ----------
        fname : str
            Path to the file where the model is stored.
        *args
            Positional arguments propagated to :meth:`~gensim.utils.SaveLoad.load`.
        **kwargs
            Key word arguments propagated to :meth:`~gensim.utils.SaveLoad.load`.


        """
        kwargs['mmap'] = kwargs.get('mmap', None)
        result = super(StmModel, cls).load(fname, *args, **kwargs)

        # check if `random_state` attribute has been set after main pickle load
        # if set -> the model to be loaded was saved using a >= 0.13.2 version of Gensim
        # if not set -> the model to be loaded was saved using a < 0.13.2 version of Gensim,
        # so set `random_state` as the default value
        if not hasattr(result, 'random_state'):
            result.random_state = gensim_utils.get_random_state(None)  # using default value `get_random_state(None)`
            logging.warning("random_state not set so using default value")

        # dtype could be absent in old models
        if not hasattr(result, 'dtype'):
            result.dtype = np.float64  # float64 was implicitly used before (cause it's default in numpy)
            logging.info("dtype was not set in saved %s file %s, assuming np.float64", result.__class__.__name__, fname)

        state_fname = gensim_utils.smart_extension(fname, '.state')
        try:
            result.state = StmState.load(state_fname, *args, **kwargs)
        except Exception as e:
            logging.warning("failed to load state from %s: %s", state_fname, e)

        id2word_fname = gensim_utils.smart_extension(fname, '.id2word')
        # check if `id2word_fname` file is present on disk
        # if present -> the model to be loaded was saved using a >= 0.13.2 version of Gensim,
        # so set `result.id2word` using the `id2word_fname` file
        # if not present -> the model to be loaded was saved using a < 0.13.2 version of Gensim,
        # so `result.id2word` already set after the main pickle load
        if os.path.isfile(id2word_fname):
            try:
                result.id2word = gensim_utils.unpickle(id2word_fname)
            except Exception as e:
                logging.warning("failed to load id2word dictionary from %s: %s", id2word_fname, e)
        return result
    
    # TODO: function comment (AND BELOW)
    def label_topics(self, topics, n, frexweight=0.5, print_labels=False):
        """
        Label topics

        Generate a set of words describing each topic from a fitted STM object.

        Highest Prob: are the words within each topic with the highest probability
        (inferred directly from topic-word distribution parameter beta)
        FREX: weights exclusivity and frequency scores to get more meaningful topic labels.
        (Bischof and Airoldi 2012 for more details.)

        @param topics number of topics to include.  Default
        is all topics.
        @param n The desired number of words (per type) used to label each topic.
        Must be 1 or greater.
        @param frexweight to control for exclusivity vs. frequency, defaults to 0.5
        @param print_labels whether labels are returned or not, defaults to False

        TODO: @return labelTopics object (list) \item{prob }{matrix of highest
        probability words}
        """
        assert n >= 1, "n must be 1 or greater"

        if topics is None:
            topics = range(self.num_topics)

        vocab = self.id2word
        wordcounts = self.wcounts

        if self.yvarlevels == None:
            frex = self.frex(w=frexweight)

            # Sort by word probabilities on each row of beta
            # Returns words with highest probability per topic
            problabels = np.argsort(-1 * self.beta)[:n]
            frexlabels = np.argsort(-1 * frex)[:n]

            out_prob = []
            out_frex = []

            for k in topics:
                probwords = [itemgetter(i)(vocab) for i in problabels[k, :n]]
                frexwords = [itemgetter(i)(vocab) for i in frexlabels[k, :n]]
                if print_labels:
                    print(f"Topic {k}:\n \t Highest Prob: {probwords}")
                    print(f"Topic {k}:\n \t FREX: {frexwords}")
                out_prob.append(probwords)
                out_frex.append(frexwords)

            return out_prob, out_frex
        else:
            labs = []
            for x in self.kappa:
                windex = np.argpartition(x, -n)[-n:]
                sorted_indices = windex[np.argsort(x[windex])][::-1]
                labs.append([vocab[i] if x[i] > 1e-5 else '' for i in sorted_indices])

            labs = np.array(labs)

            anames = self.yvarlevels
            i1, i2 = self.num_topics, self.num_topics + len(self.yvarlevels)
            out_topics = labs[topics]
            out_covariate = labs[i1:i2]

            if self.interactions:
                intnums = np.arange(i2, labs.shape[0])
                tindx = np.repeat(np.arange(0, self.num_topics), len(self.yvarlevels))
                filtered_intnums = intnums[np.isin(tindx, topics)]
                out_interaction = labs[filtered_intnums, :]

            topiclabs = ["Topic Words:\n"]
            topiclabs.extend([f"Topic {topic}: {', '.join(map(str, row))}\n" for topic, row in enumerate(out_topics)])

            aspects = ["Covariate Words:\n"]
            aspects.extend([f"Group {level}: {', '.join(map(str, row))}\n" for level, row in zip(anames, out_covariate)])

            if self.interactions:
                interactions = ["Topic-Covariate Interactions:\n"]
                intlabs = np.array([", ".join(row) for row in out_interaction])
                topicnums = np.concatenate([np.tile(topics, len(anames))])

                for i in topics:
                    topic_terms = intlabs[topicnums == i]
                    for a, aspect_term in zip(anames, topic_terms):
                        out = f"Topic {i}, Group {a}: {aspect_term} \n"
                        interactions.append(out)

                labels = topiclabs + ["\n"] + aspects + ["\n"] + interactions
                print("".join(labels))
                return out_topics, out_covariate, out_interaction
            
            labels = topiclabs + ["\n"] + aspects
            print("".join(labels))
            return out_topics, out_covariate, None

    def frex(self, w=0.5):
        """Calculate FREX (FRequency and EXclusivity) words
        A primarily internal function for calculating FREX words.
        Exclusivity is calculated by column-normalizing the beta matrix (thus representing the conditional probability of seeing
        the topic given the word).  Then the empirical CDF of the word is computed within the topic.  Thus words with
        high values are those where most of the mass for that word is assigned to the given topic.

        @param logbeta a K by V matrix containing the log probabilities of seeing word v conditional on topic k
        @param w a value between 0 and 1 indicating the proportion of the weight assigned to frequency

        """
        beta = np.log(self.beta)
        log_exclusivity = beta - logsumexp(beta, axis=0)
        exclusivity_ecdf = np.apply_along_axis(self.ecdf, 1, log_exclusivity)
        freq_ecdf = np.apply_along_axis(self.ecdf, 1, beta)
        out = 1.0 / (w / exclusivity_ecdf + (1 - w) / freq_ecdf)
        return out

    def find_thoughts(self, topics=None, threshold=0, n=3):
        """
        Return the most prominent documents for a certain topic in order to identify representative
        documents. Topic representing documents might be conclusive underlying structure in the text
        collection.
        Following Roberts et al. (2016b):
        Theta captures the modal estimate of the proportion of word
        tokens assigned to the topic under the model.

        @param: threshold (np.float) minimal theta value of the documents topic proportion
            to be taken into account for the return statement.
        @param: topics to get the representative documents for
        @return: the top n document indices ranked by the MAP estimate of the topic's theta value

        Example: Return the 10 most representative documents for the third topic:
        > data.iloc[model.find_thoughts(topics=[3], n=10)]

        """
        assert n > 1, "Must request at least one returned document"
        if n > self.lencorpus:
            n = self.lencorpus

        if topics is None:
            topics = range(self.num_topics)

        thoughts = []
        for k in topics:
            # grab the values and the rank
            index = np.argsort(-1 * self.theta[:, k])[1:n]
            val = -np.sort(-1 * self.theta[:, k])[1:n]
            # subset to those values which meet the threshold
            index = index[np.where(val >= threshold)]
            # grab the document(s) corresponding to topic k
            thoughts.append(index)
        return thoughts

    def ecdf(self, arr):
        """Calculate the ECDF values for all elements in a 1D array."""
        return rankdata(arr, method="max") / arr.size