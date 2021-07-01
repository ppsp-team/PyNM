import re
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects import r

class GAMLSS:
    def __init__(self,mu=None,sigma=None,nu=None,tau=None,family='SHASHo2',lib_loc=None):
        numpy2ri.activate()
        pandas2ri.activate()

        if lib_loc is None:
            self.gamlss_data = importr('gamlss.data')
            self.gamlss_dist = importr('gamlss.dist')
            self.gamlss = importr('gamlss')
        else:
            self.gamlss_data = importr('gamlss.data',lib_loc=lib_loc)
            self.gamlss_dist = importr('gamlss.dist',lib_loc=lib_loc)
            self.gamlss = importr('gamlss',lib_loc=lib_loc)
        
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
            Formula for mu (location) parameter of GAMLSS model.
        sigma: str or None
            Formula for mu (location) parameter of GAMLSS model.
        nu: str or None
            Formula for mu (location) parameter of GAMLSS model.
        tau: str or None
            Formula for mu (location) parameter of GAMLSS model.

        Raises
        ------
        ValueError
            If any of the input strings contains a function call not recognised by the R GAMLSS package.
        
        Returns
        -------
        R formula, R formula, R formula, R formula
            R formula equivalent for each input string.
        """
        #TODO: convert None to R
        if mu is None:
            mu = '{} ~ {}'.format(self.score,'+'.join(self.confounds))
        if sigma is None:
            sigma = '~ 1'
        if nu is None:
            nu = '~ 1'
        if tau is None:
            tau = '~ 1'
        
        #from formulas get r function
        p = re.compile("\w*\(")
        funcs = []
        for s in [mu,sigma,nu,tau]: #TODO: when convert None, update to list of existing
            for f in p.findall(s):
                funcs.append(f[:-1])

        for func in funcs:
            try:
                exec("{} = r['{}']".format(func,func))
            except:
                raise ValueError("'{}' function not found in R GAMLSS package. See GAMLSS documentation for available functions.".format(func))

        formula = r['formula']
        mu_f = formula(mu)
        sigma_f = formula(sigma)
        nu_f = formula(nu)
        tau_f = formula(tau)
        return mu_f,sigma_f,nu_f,tau_f
    
    def fit(self,train_data):
        self.model = self.gamlss.gamlss(self.mu_f,
                    sigma_formula=self.sigma_f,
                    nu_formula=self.nu_f,
                    tau_formula=self.tau_f,
                    family=self.rfamily,
                    data=train_data)
    
    def predict(self,test_data):
        res = self.gamlss.predict_gamlss(self.model,newdata=test_data)
        return res