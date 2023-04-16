class Covariates:
    """
    Class to store covariates.
    """
    def __init__(self, prevalence, betaindex, yvarlevels, formula):
        """
        Constructor.
        :param prevalence: covariate matrix
        :param betaindex: beta index
        :param yvarlevels: y variable levels
        :param formula: formula
        """
        self.prevalence = prevalence
        self.beta_index = betaindex
        self.y_var_levels = yvarlevels
        self.formula = formula