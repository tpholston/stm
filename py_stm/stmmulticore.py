import logging
import queue
import time
from qpsolvers import solve_qp
from multiprocessing import Pool, Queue, cpu_count

import numpy as np
import gensim.utils as gensim_utils

from . import StmModel, StmState, utils


logger = logging.getLogger(__name__)

class StmMulticore(StmModel):
	"""
	Multicore implementation of the STM model.
	"""
	def __init__(self, corpus=None, num_topics=100, id2word=None, workers=None,
                 metadata=None, prevalence=None, content=None, chunksize=2000, 
         	     passes=1, batch=False, decay=0.5, offset=1.0, 
         	     eval_every=10, minimum_probability=0.01, random_state=None, 
         	     minimum_phi_value=0.01,  per_word_topics=False, callbacks=None, 
         	     dtype=np.float32, init_mode="Spectral", max_vocab=5000, 
         	     interactions=True, convergence_threshold=1e-5, LDAbeta=True, 
         	     sigma_prior=0, model="STM", gamma_prior="OLS"):
		"""
		
		Parameters
		----------
		corpus : TODO:
		"""
		self.workers = max(1, cpu_count() - 1) if workers is None else workers
		self.batch = batch
		
		super(StmMulticore, self).__init__(
			corpus=corpus, num_topics=num_topics, id2word=id2word, 
			metadata=metadata, prevalence=prevalence, content=content,
			chunksize=chunksize, passes=passes, decay=decay, offset=offset, 
			eval_every=eval_every, minimum_probability=minimum_probability, 
			random_state=random_state, minimum_phi_value=minimum_phi_value, 
			per_word_topics=per_word_topics, callbacks=callbacks, dtype=dtype, 
			init_mode=init_mode, max_vocab=max_vocab, interactions=interactions, 
			convergence_threshold=convergence_threshold, LDAbeta=LDAbeta, 
			sigma_prior=sigma_prior, model=model, gamma_prior=gamma_prior
		)
	
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

		M = gram_matrix[np.int64(anchor_words)]
		P = np.dot(M, M.T).toarray()

		G = np.eye(M.shape[0])
		h = np.zeros(M.shape[0])
		
		args_list = [(gram_matrix, anchor_words, word_probabilities, i, M, P, G, h) for i in range(gram_matrix.shape[0])]

		with Pool() as pool:
			results = pool.map(recover_l2_helper, args_list)

		weights = np.vstack(results)
		A = weights.T * word_probabilities
		A = A.T / np.sum(A, axis=1)

		assert np.any(A > 0), "Negative probabilities for some words."
		assert np.any(A < 1), "Word probabilities larger than one."

		recovered_beta = A.T
		if top_words_indices is not None:
			updated_beta = np.zeros(self.num_topics * len(self.id2word)).reshape(self.num_topics, len(self.id2word))
			updated_beta[:, top_words_indices] = recovered_beta
			updated_beta += 0.001 / len(self.id2word)
			recovered_beta = updated_beta / np.sum(updated_beta)

		# Create a list of beta values for each aspect
		if self.interactions:
			recovered_beta = np.array([recovered_beta.copy() for _ in set(self.betaindex)])

		return recovered_beta
	
	def update(self, corpus, chunks_as_numpy=False):
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
		try:
			self.lencorpus = len(corpus)
		except Exception:
			logger.warning("input corpus stream has no len(); counting documents")
			self.lencorpus = sum(1 for _ in corpus)
		if self.lencorpus == 0:
			logger.warning("StmModel.update() called with an empty corpus")
			return
		
		if self.batch:
			updatetype = "batch"
			updateafter = self.lencorpus
		else:
			updatetype = "online"
			updateafter = self.chunksize * self.workers
		eval_every = self.eval_every or 0
		evalafter = min(self.lencorpus, eval_every * updateafter)

		updates_per_pass = max(1, self.lencorpus / updateafter)

		logger.info(
			"running %s STM training, %s topics, %i passes over "
			"the supplied corpus of %i documents, updating model once "
			"every %i documents, evaluating perplexity every %i documents, "
			"with a convergence threshold of %f",
			updatetype, self.num_topics, self.passes, self.lencorpus,
			updateafter, evalafter, self.convergence_threshold
		)

		if updates_per_pass * self.passes < 10:
			logger.warning(
				"too few updates, training might not converge; "
				"consider increasing the number of passes or iterations to improve accuracy"
			)
		
		job_queue = Queue(maxsize=2 * self.workers)
		result_queue = Queue()

		def process_result_queue(force=False):
			"""
			Clear the result queue, merging all intermediate results, and update the
			STM model if necessary.

			"""
			#logger.warning(f"Merging for pass: {pass_}")
			merged_new = False
			while not result_queue.empty():
				other.merge(result_queue.get())
				queue_size[0] -= 1
				merged_new = True

			if (force and merged_new and queue_size[0] == 0):
				self.bound = np.sum(other.calculated_bounds)
				self.last_bounds.append(self.bound)
				self.calculated_bounds = [] # reset bounds

				elapsed_time = np.round((time.time() - start_estep), 3)

				logger.info(f"Lower Bound: {self.bound}")
				logger.info(f"Completed E-Step in {elapsed_time} seconds. \n")

				self.do_mstep(other, pass_ > 0)
				logger.warning(f"PASS {pass_}, {self.state.numdocs}")
				# if eval_every > 0 and (force or (self.num_updates / updateafter) % eval_every == 0):
				# 	self.log_perplexity(chunk, total_docs=self.lencorpus) # TODO: lencorpus log_perplexity
 
		logger.info("training STM model using %i processes", self.workers)
		pool = Pool(self.workers, do_worker_estep, (job_queue, result_queue, self))

		first_start_time = time.time()
		for pass_ in range(self.passes):
			other = StmState(self.beta, self.sigma, self.mu, self.theta_shape, self.lambda_shape, 
                             self.beta.shape, self.sigma.shape, dtype=self.dtype)
			self.state.reset()
			queue_size, reallen = [0], 0
			chunk_stream = gensim_utils.grouper(corpus, self.chunksize, as_numpy=chunks_as_numpy, dtype=self.dtype)

			start_estep = time.time()
			for chunk_no, chunk in enumerate(chunk_stream):
				reallen += len(chunk)  # keep track of how many documents we've processed so far

				chunk_indices = range(chunk_no * self.chunksize, min(chunk_no * self.chunksize + len(chunk), self.lencorpus))
		
				# put the chunk into the workers' input job queue
				while True:
					try:
						job_queue.put((chunk_no, chunk, chunk_indices, self.state), block=False)
						queue_size[0] += 1

						logger.info(
							"PROGRESS: pass %i, dispatched chunk #%i = documents up to #%i/%i, "
							"outstanding queue size %i",
							pass_, chunk_no, chunk_no * self.chunksize + len(chunk), self.lencorpus, queue_size[0]
						)
						break
					except queue.Full:
						# in case the input job queue is full, keep clearing the
						# result queue, to make sure we don't deadlock
						process_result_queue() # TODO: go back to this

				process_result_queue()
			
			while queue_size[0] > 0:
				process_result_queue(force=True)  

			if reallen != self.lencorpus:
				raise RuntimeError("input corpus size changed during training (don't use generators as input)")
			
		pool.terminate()

		self.time_processed = time.time() - first_start_time
		logger.warning(
			f"({self.passes}) iterations conducted in {self.time_processed} seconds"
		)

def do_worker_estep(input_queue, result_queue, worker_stm):
	"""Perform E-step for each job.

	Parameters
	----------
	input_queue : queue of (int, list of (int, float), :class:`~py_stm.stmmulticore.StmMulticore`)
		Each element is a job characterized by its ID, the corpus chunk to be processed in BOW format and the worker
		responsible for processing it.
	result_queue : queue of :class:`~py_stm.stm.StmState`
		After the worker finished the job, the state of the resulting (trained) worker model is appended to this queue.
	worker_stm : :class:`~py_stm.stmmulticore.StmMulticore`
		STM instance which performed e step
	"""
	logger.debug("worker process entering E-step loop")
	while True:
		logger.debug("getting a new job")
		chunk_no, chunk, chunk_indices, w_state = input_queue.get()
		logger.debug("processing chunk #%i of %i documents", chunk_no, len(chunk))
		worker_stm.state = w_state
		worker_stm.state.reset()
		worker_stm.do_estep(chunk, chunk_indices)
		#logger.warning(worker_stm.state.sigma_ss)
		#logger.warning(worker_stm.sigma)
		del chunk
		logger.debug("processed chunk, queuing the result")
		result_queue.put(worker_stm.state)
		worker_stm.state = None
		logger.debug("result put")

def recover_l2_helper(args):
    Qbar, anchors, word_probabilities, i, M, P, G, h = args
    
    if i in anchors:
        vec = np.repeat(0, P.shape[0])
        vec[np.where(anchors == i)] = 1
        condprob = vec
    else:
        y = Qbar[i]
        q = (M @ y.T).toarray().flatten()
        solution = solve_qp(P=P, q=q, G=G, h=h, solver='quadprog')
        condprob = -1 * solution

    return condprob