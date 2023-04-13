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
		raise ValueError(f"internal id out of bounds ({val}, expected <0..{self.num}))")

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