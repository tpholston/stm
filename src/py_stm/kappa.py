class Kappa:
    """
    Class to store gamma related variables.
    """
    def __init__(self, lda_beta, interactions, fixed_intercept=True, mstep_tol=0.001, mstep_maxit=3, contrast=False):
        """
        Constructor
            :param lda_beta: lda beta bool
            :param interactions: interactions bool
            :param fixed_intercept: TODO: bool
            :param mstep_tol: TODO:
            :param mstep_maxit: max iterations    
            :param contrast: TODO: bool
        """
        self.lda_beta = lda_beta
        self.interactions = interactions
        self.fixed_intercept = fixed_intercept
        self.mstep_tol = mstep_tol
        self.mstep_maxit = mstep_maxit
        self.contrast = contrast