import re
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects import r

class GAMLSS:
    """Class for GAMLSS model.
        
        Attributes
        ----------
        gamlss_data: R package
            Python imported R package.
        gamlss_dist: R package
            Python imported R package.
        gamlss: R package
            Python imported R package.
        mu_f: R formula
            Formula for mu (location) parameter.
        sigma_f: R formula
            Formula for sigma (shape) parameter.
        nu_f: R formula
            Formula for nu parameter.
        tau_f: R formula
            Formula for tau parameter.
        rfamily: R object
            Family of distributions to use for fitting.
        model: R object
            Fitted gamlss model.
        """

    def __init__(self,mu=None,sigma=None,nu=None,tau=None,family='SHASHo2',what='mu',lib_loc=None,score=None,confounds=None):
        """Create GAMLSS object. Formulas must be written for R, using functions available in the GAMLSS package.
        
        Parameters
        ----------
        mu: str, default=None
            Formula for mu (location) parameter. If None, formula for score is sum of confounds (must have both score and confounds not None).
        sigma: str, default=None
            Formula for sigma (shape) parameter. If None, formula is '~ 1'.
        nu: str, default=None
            Formula for nu parameter. If None, formula is '~ 1'.
        tau: str, default=None
            Formula for tau parameter. If None, formula is '~ 1'.
        family: str,default='SHASHo2'
            Family of distributions to use for fitting, default is 'SHASHo2'. See R documentation for GAMLSS package for other available families of distributions.
        what: str, default='mu'
            What parameter to predict, can be 'mu', 'sigma', 'nu' or 'tau'.
        lib_loc: str, default=None
            Path to location of installed GAMLSS package.
        score: str, default=None
            Label of score in DataFrame.
        confounds: list, default=None
            List of labels of confounds in DataFrame.
        """
        numpy2ri.activate()
        pandas2ri.activate()

        if lib_loc is None:
            self.gamlss_data = importr('gamlss.data')
            self.gamlss_dist = importr('gamlss.dist')
            self.gamlss = importr('gamlss')
            self.base = importr('base')
        else:
            self.gamlss_data = importr('gamlss.data',lib_loc=lib_loc)
            self.gamlss_dist = importr('gamlss.dist',lib_loc=lib_loc)
            self.gamlss = importr('gamlss',lib_loc=lib_loc)
        
        self.what = what
        self.score = score
        self.confounds = confounds
        self.mu_f,self.sigma_f,self.nu_f,self.tau_f = self._get_r_formulas(mu,sigma,nu,tau)
        try:
            self.rfamily = r[family]
        except:
            raise ValueError("Provided family not valid, choose 'SHASHo2', 'NO' or see R documentation for GAMLSS package for other available families of distributions.")
    
    def _get_r_formulas(self,mu,sigma,nu,tau):
        """Convert from string input to R formula.

        Parameters
        ----------
        mu: str or None
            Formula for mu (location) parameter of GAMLSS model. If None, formula for score is sum of confounds.
        sigma: str or None
            Formula for mu (location) parameter of GAMLSS model. If None, formula is '~ 1'.
        nu: str or None
            Formula for mu (location) parameter of GAMLSS model. If None, formula is '~ 1'.
        tau: str or None
            Formula for mu (location) parameter of GAMLSS model. If None, formula is '~ 1'.

        Raises
        ------
        ValueError
            If any of the input strings contains a function call not recognised by the R GAMLSS package.
        ValueError
            If mu is None and either score or confounds is None.
        
        Returns
        -------
        R formula, R formula, R formula, R formula
            R formula equivalent for each input string.
        """
        if mu is None:
            if (self.score is None) or (self.confounds is None):
                raise ValueError('If mu is None, both score and confounds must be provided i.e. not None.')
            mu = '{} ~ {}'.format(self.score,' + '.join(self.confounds))
        if sigma is None:
            sigma = '~ 1'
        if nu is None:
            nu = '~ 1'
        if tau is None:
            tau = '~ 1'
        
        # get r functions from formulas
        p = re.compile(r"\w*\(") # raw string (to avoid deprecation warning)
        funcs = []
        for s in [mu,sigma,nu,tau]:
            for f in p.findall(s):
                funcs.append(f[:-1])

        for func in funcs:
            try:
                exec("{} = r['{}']".format(func,func))
            except:
                raise ValueError("'{}' function not found in R GAMLSS package. See GAMLSS documentation for available functions.".format(func))

        formula = r['formula']
        return formula(mu),formula(sigma),formula(nu),formula(tau)
    
    def fit(self,train_data):
        """Create and fit gamlss model.

        Parameters
        ----------
        train_data: DataFrame
            DataFrame with training data.
        """
        self.model = self.gamlss.gamlss(self.mu_f,
                    sigma_formula=self.sigma_f,
                    nu_formula=self.nu_f,
                    tau_formula=self.tau_f,
                    family=self.rfamily,
                    data=train_data)
    
    def predict(self,test_data):
        """Predict from fitted gamlss model.
        
        Parameters
        ----------
        test_data: DataFrame
            DataFrame with test data.
        what: str
            Which parameter to predict, can be 'mu','sigma', 'nu', or 'tau'.
        """
        res = self.gamlss.predict_gamlss(self.model,newdata=test_data,what=self.what)
        return res
