class Gamma:
    """
    Class to store gamma related variables.
    """
    def __init__(self, mode, prior=None, enet=1, ic_k=2, max_its=1000):
        """
        Constructor.
		:param mode: gamma_prior
		:param prior: TODO: not sure
		:param enet: TODO: not sure
		:param ic_k: TODO: not sure
		:param max_its: maximum iterations
        """
        self.mode = mode
        self.prior = prior
        self.enet = enet
        self.ic_k = ic_k
        self.max_its = max_its