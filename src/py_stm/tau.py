class Tau:
    """
    Class to store tau related variables
    """
    def __init__(self, mode, tol=1e-5, enet=1, nlambda=250, lambda_min_ratio=0.001, ic_k=2, maxit=1e4):
        """
        Constructor
            :param mode: kappa prior
            :param tol: TODO: float
            :param enet: TODO:
            :param nlambda: lambda number
            :param lambda_min_ratio: minimum lambda ratio
            :param ic_k: TODO:
            :param maxit: max iterations
	    """
        self.mode = mode
        self.tol = tol
        self.enet = enet
        self.nlambda = nlambda
        self.lambda_min_ratio = lambda_min_ratio
        self.ic_k = ic_k
        self.maxit = maxit