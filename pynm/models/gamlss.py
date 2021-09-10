import re
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects import r
from pynm.util import read_confounds

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
            Formula for sigma (scale) parameter.
        nu_f: R formula
            Formula for nu (skewness) parameter.
        tau_f: R formula
            Formula for tau (kurtosis) parameter.
        rfamily: R object
            Family of distributions to use for fitting.
        method: str
            Method to fit GAMLSS.
        model: R object
            Fitted GAMLSS model.
        """

    def __init__(self,mu=None,sigma=None,nu=None,tau=None,family='SHASHo2',method='RS',score=None,confounds=None):
        """Create GAMLSS object. Formulas must be written for R, using functions available in the GAMLSS package.
        
        Parameters
        ----------
        mu: str, default=None
            Formula for mu (location) parameter of GAMLSS. If None, formula for score is sum of confounds
            with non-categorical columns as smooth functions, e.g. "score ~ ps(age) + sex".
        sigma: str, default=None
            Formula for sigma (scale) parameter of GAMLSS. If None, formula is '~ 1'.
        nu: str, default=None
            Formula for nu (skewness) parameter of GAMLSS. If None, formula is '~ 1'.
        tau: str, default=None
            Formula for tau (kurtosis) parameter of GAMLSS. If None, formula is '~ 1'.
        family: str,default='SHASHo2'
            Family of distributions to use for fitting, default is 'SHASHo2'. See R documentation for GAMLSS package for other available families of distributions.
        method: str, default = 'RS'
            Method for fitting GAMLSS. Can be 'RS' (Rigby and Stasinopoulos algorithm), 'CG' (Cole and Green algorithm) or 'mixed(n,m)' where n & m are integers.
            Specifying 'mixed(n,m)' will use the RS algorithm for n iterations and the CG algorithm for up to m additional iterations.
        score: str, default=None
            Label of score in DataFrame.
        confounds: list, default=None
            List of labels of confounds in DataFrame.
        
        Notes
        -----
        If using 'random()' to model a random effect in any of the formulas, it must be passed a column of the dataframe with categorical values
        as a factor: e.g. 'random(as.factor(COL))'.
        """
        numpy2ri.activate()
        pandas2ri.activate()

        self.gamlss_data = importr('gamlss.data')
        self.gamlss_dist = importr('gamlss.dist')
        self.gamlss = importr('gamlss')
        self.base = importr('base')
        
        self.score = score
        self.confounds = confounds
        self.mu_f,self.sigma_f,self.nu_f,self.tau_f = self._get_r_formulas(mu,sigma,nu,tau)
        self.family = family
        self.method = self._get_method(method)
        try:
            self.rfamily = r[family]
        except:
            raise ValueError("Provided family not valid, choose 'SHASHo2', 'NO' or see R documentation for GAMLSS package for other available families of distributions.")
    
    def _get_r_formulas(self,mu,sigma,nu,tau):
        """Convert from string input to R formula.

        Parameters
        ----------
        mu: str or None
            Formula for mu (location) parameter of GAMLSS. If None, formula for score is sum of confounds
            with non-categorical columns as smooth functions, e.g. "score ~ ps(age) + sex".
        sigma: str or None
            Formula for sigma (scale) parameter of GAMLSS. If None, formula is '~ 1'.
        nu: str or None
            Formula for nu (skewness) parameter of GAMLSS. If None, formula is '~ 1'.
        tau: str or None
            Formula for tau (kurtosis) parameter of GAMLSS. If None, formula is '~ 1'.

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
            _,cat = read_confounds(self.confounds)
            formula_conf = ['ps({})'.format(conf) for conf in self.confounds if not conf[2:-1] in cat] + cat
            mu = '{} ~ {}'.format(self.score,' + '.join(formula_conf))
        if sigma is None:
            sigma = '~ 1'
        if nu is None:
            nu = '~ 1'
        if tau is None:
            tau = '~ 1'
        
        # get r functions from formulas
        p = re.compile(r"\w*\(")
        funcs = []
        for s in [mu,sigma,nu,tau]:
            for f in p.findall(s):
                funcs.append(f[:-1])

        for func in funcs:
            try:
                exec("{} = r['{}']".format(func,func))
            except:
                raise ValueError("'{}' function not found in R GAMLSS package. See GAMLSS documentation for available functions.".format(func))

        return mu,sigma,nu,tau
    
    def _get_method(self,method):
        """ Get method parameter in appropriate format for R.

        Raises
        ------
        TypeError
            "Argument 'method' must be of type str."
        ValueError
            "Unrecognized argument for 'method'."
        """
        if not isinstance(method,str):
            raise TypeError("Argument 'method' must be of type str.")

        pattern = re.compile(r"mixed\([0-9]*,[0-9]*\)")
    
        if method == 'RS':
            return 'RS()'
        elif method == 'CG':
            return 'CG()'
        elif pattern.match(method) is not None:
            return method
        else:
            raise ValueError("Unrecognized argument for 'method'.")
    
    def fit(self,train_data):
        """Create and fit gamlss model.

        Parameters
        ----------
        train_data: DataFrame
            DataFrame with training data.
        """
        ro.globalenv['train_data'] = train_data

        self.model = r(f'''gamlss({self.mu_f},
                                sigma.formula={self.sigma_f},
                                nu.formula={self.nu_f},
                                tau.formula={self.tau_f},
                                family={self.family},
                                data=train_data,
                                method={self.method})''')
                    
    def predict(self,test_data,what='mu'):
        """Predict from fitted gamlss model.
        
        Parameters
        ----------
        test_data: DataFrame
            DataFrame with test data.
        what: str
            Which parameter to predict, can be 'mu','sigma', 'nu', or 'tau'.
        """
        ro.globalenv['model'] = self.model
        ro.globalenv['test_data'] = test_data

        res = r(f'''predict(model,newdata=test_data,parameter="{what}")''')
        return res
