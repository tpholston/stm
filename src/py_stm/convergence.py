class Convergence:
    """
    Class to store convergence variables
    """
    def __init__(self, max_em_its, emtol, allow_negative_change):
        """
        Parameters
        ----------
        max_em_its : int
            Maximum number of EM iterations
        emtol : float
            EM convergence tolerance
        allow_negative_change : bool
            Whether to allow negative changes in EM
        """
        assert max_em_its > 0, "max_em_its must be positive"
        assert emtol > 0, "emtol must be positive"
        assert allow_negative_change in [True, False], "allow_negative_change must be True or False"
        
        self.max_em_its = max_em_its
        self.emtol= emtol
        self.allow_negative_change = allow_negative_change