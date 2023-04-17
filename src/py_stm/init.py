class Init:
    """
    class to store 'init' related variables. Note that this is not the __init__ file
    """
    def __init__(self, mode, alpha, nits=50, burnin=25, eta=0.01, s=0.05, p=3000, d_group_size=2000, recoverEG=True, tSNE_init_dims=50, tSNE_perplexity=30):
        """
        Constructor
            :param mode: init mode
            :param alpha: TODO:
            :param nits: number of iterations
            :param burnin: TODO:
            :param eta: TODO:
            :param s: TODO:
            :param p: TODO:
            :param d_group_size: TODO:
            :param recoverEG: TODO:
            :param tSNE_init_dims: TODO:
            :param tSNE_perplexity: TODO:
        """
        self.mode = mode
        self.alpha = alpha
        self.nits = nits
        self.burnin = burnin
        self.eta = eta
        self.s = s
        self.p = p
        self.d_group_size = d_group_size
        self.recoverEG = recoverEG
        self.tSNE_init_dims = tSNE_init_dims
        self.tSNE_perplexity = tSNE_perplexity
        self.max_v = 100 # TODO: may need to modify this
        self.custom = "" # TODO: may need to set this