#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : PyNM.py
# description     : Gaussian Processes, Centiles & LOESS-based normative models
# author          : Guillaume Dumas (Institut Pasteur/Université de Montréal)
#                   Annabelle Harvey (Université de Montréal)
# date            : 2021-04-15
# notes           : The input dataframe column passed to --group must either
#                   have controls marked as "CTR" and probands as "PROB", or
#                   controls marked as 0 and probands as 1.
#                   The --pheno_p is for the path to the input dataframe.
#                   The --out_p flag is for the path to save the output
#                   dataframe, including filename formatted as 'filename.csv'.
#                   The confounds columns for the gaussian process model must
#                   be specified using the --confounds flag. The confound for
#                   the LOESS and centiles models must be specified using the
#                   --conf flag.
# licence         : BSD 3-Clause License
# python_version  : 3.7
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.diagnostic import het_white
from statsmodels.tools.tools import add_constant
from scipy.stats.mstats import mquantiles
from scipy import stats

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


def _read_confounds(confounds):
    """ Process input list of confounds.

    Parameters
    ----------
    confounds : list of str
        List of confounds with categorical variables indicated by c(var) ('c' must be lower case).

    Returns
    -------
    list
        List of all confounds without wrapper on categorical variables: c(var) -> var.
    list
        List of only categorical confounds without wrapper.
    """
    categorical = []
    clean_confounds = []
    for conf in confounds:
        if ((conf[0:2] == 'c(') & (conf[-1] == ')')):
            categorical.append(conf[2:-1])
            clean_confounds.append(conf[2:-1])
        else:
            clean_confounds.append(conf)
    return clean_confounds, categorical


class PyNM:
    """ Class to run normative modeling using LOESS, centiles, or GP model.

    Attributes
    ----------
    data : dataframe
        Dataset to fit model, must at least contain columns corresponding to 'group',
        'score', and 'conf'.
    score : str
        Label of column from data with score (response variable).
    group : str
        Label of column from data that encodes wether subjects are probands or controls.
    CTR : str or int
        Label of controls in 'group' column can be 'CTR' or 0.
    PROB: str or int
        Label of probands in 'group' column can be 'PRB' or 1.
    conf: str
        Label of column from data with confound to use for LOESS and centiles models.
    confounds: list of str
        List of labels of columns from data with confounds. For GP model all confounds will be used,
        for LOESS and Centiles models only the first is used. For GAMLSS all confounds are used
        unless formulas are specified. Categorical values must be denoted by c(var) ('c' must be lower case).
    train_sample: str or float
        Which method to use for a training sample, can be 'controls' to use all the controls, 
        'manual' to be manually set, or a float in (0,1] for a percentage of controls.
    bin_spacing: int
        Distance between bins for LOESS & centiles models.
    bin_width: float
        Width of bins for LOESS & centiles models.
    bins: array
        Bins for the centiles and LOESS models.
    bin_count: array
        Number of controls in each bin.
    zm: array
        Mean of each bin.
    zstd: array
        Standard deviation of each bin.
    zci: array
        Confidence interval of each bin.
    z: array
        Centiles for each bin.
    SMSE_LOESS: float
        Mean Square Error of LOESS normative model
    SMSE_Centiles: float
        Median Square Error of Centiles normative model
    SMSE_GP: float
        Mean Square Error of Gaussian Process normative model
    MSLL: float
        Mean Standardized Log Loss of Gaussian Process normative model
    """

    def __init__(self, data, score='score', group='group', confounds=['age', 'c(sex)', 'c(site)'], train_sample='controls',
                bin_spacing=-1, bin_width=-1):
        """ Create a PyNM object.

        Parameters
        ----------
        data : dataframe
            Dataset to fit model, must at least contain columns corresponding to 'group',
            'score', and 'conf'.
        score : str, default='score'
            Label of column from data with score (response variable).
        group : str, default='group'
            Label of column from data that encodes wether subjects are probands or controls.
        confounds: list of str, default=['age', 'c(sex)', 'c(site)']
            List of labels of columns from data with confounds. For GP model all confounds will be used,
            for LOESS and Centiles models only the first is used. For GAMLSS all confounds are used
            unless formulas are specified. Categorical values must be denoted by c(var) ('c' must be lower case).
        train_sample: str or float, default='controls'
            Which method to use for a training sample, can be 'controls' to use all the controls, 
            'manual' to be manually set, or a float in (0,1] for a percentage of controls.
        bin_spacing: int, default=-1
            Distance between bins for LOESS & centiles models.
        bin_width: float, default=-1
            Width of bins for LOESS & centiles models.
        """
        self.data = data.copy()
        self.score = score
        self.group = group
        self.confounds = confounds
        self.conf = self.confounds[0]
        self.train_sample = train_sample
        self.CTR = None
        self.PROB = None
        self.bin_spacing = bin_spacing
        self.bin_width = bin_width
        self.bins = None
        self.bin_count = None
        self.zm = None
        self.zstd = None
        self.zci = None
        self.z = None
        self.SMSE_LOESS = None
        self.SMSE_Centiles = None
        self.SMSE_GP = None
        self.MSLL = None

        self._set_group_names()
        self._set_group()

    def _make_train_sample(self, train_size):
        """ Select a subsample of controls to be used as a training sample for the normative model.

        Parameters
        ----------
        train_size: float
            Percentage of controls to use for training. Must be in (0,1].
        """
        ctr_idx = self.data[self.data[self.group] == self.CTR].index.tolist()
        n_ctr = len(ctr_idx)
        n_ctr_train = max(int(train_size*n_ctr), 1)  #TODO: make this minimum 2?

        np.random.seed(1)
        ctr_idx_train = np.array(np.random.choice(ctr_idx, size=n_ctr_train, replace=False))
        
        train_sample = np.zeros(self.data.shape[0])
        train_sample[ctr_idx_train] = 1
        self.data['train_sample'] = train_sample

        print('Models will be fit with train sample size = {}: using {}/{} of controls.'.format(train_size, n_ctr_train, n_ctr))

    def _set_group(self):
        """ Read the specified training sample and set the group attribute to refer to the appropriate column of data.

        Raises
        ------
        ValueError
            With train_sample='contols': Dataset has no controls for training sample.
        ValueError
            With train_sample='manual': Data has no column "train_sample". To manually specify a training sample, 
            data .csv must contain a column "train_sample" with included subjects marked with 1 and rest as 0.
        ValueError
            With train_sample='manual': Dataset has no subjects in specified training sample.
        ValueError
            Value for train_sample not recognized. Must be either 'controls', 'manual', or a value in (0,1].
        ValueError
            With train_sample float: Numerical value for train_sample must be in the range (0,1].
        """
        if self.train_sample == 'controls':
            print('Models will be fit on full set of controls.')
            if self.data[self.data[self.group] == self.CTR].shape[0] == 0:
                raise ValueError('Dataset has no controls for training sample.')
        elif self.train_sample == 'manual':
            print('Models will be fit using specified training sample.')
            if 'train_sample' not in self.data.columns:
                raise ValueError('Data has no column "train_sample". To manually specify a training sample, data .csv '
                                 'must contain a column "train_sample" with included subjects marked with 1 and rest as 0.')
            self.group = 'train_sample'
            self._set_group_names()

            if self.data[self.data[self.group] == self.CTR].shape[0] == 0:
                raise ValueError('Dataset has no subjects in specified training sample..')
        else:
            try:
                train_size = float(self.train_sample)
            except:
                raise ValueError("Value for train_sample not recognized. Must be either 'controls', 'manual', or a "
                                 "value in (0,1].")
            else:
                if (train_size > 1) or (train_size <= 0):
                    raise ValueError("Numerical value for train_sample must be in the range (0,1].")
                else:
                    self._make_train_sample(train_size)
                    self.group = 'train_sample'
                    self._set_group_names()

    def _set_group_names(self):
        """ Read whether subjects in data are labeled CTR/PROB or 0/1 and set labels accordingly."""
        if self.group == 'train_sample':
            self.CTR = 1
            self.PROB = 0
        else:
            labels = list(self.data[self.group].unique())
            if ('CTR' in labels) or ('PROB' in labels):
                self.CTR = 'CTR'
                self.PROB = 'PROB'
            else:
                self.CTR = 0
                self.PROB = 1

    def _get_masks(self):
        """ Get masks from data corresponding to controls and probands.

        Returns
        -------
        array
            Control mask: controls marked as True.
        array
            Proband mask: probands marked as True.
        """
        ctr = self.data.loc[(self.data[self.group] == self.CTR)]
        ctr_mask = self.data.index.isin(ctr.index)
        probands = self.data.loc[(self.data[self.group] == self.PROB)]
        prob_mask = self.data.index.isin(probands.index)
        return ctr_mask, prob_mask

    # Default values for age in days
    def _create_bins(self):
        """ Create bins for the centiles and LOESS models.
        Returns
        -------
        array
            Bins for the centiles and LOESS models.
        """
        min_conf = self.data[self.conf].min()
        max_conf = self.data[self.conf].max()

        if self.bin_width == -1:
            self.bin_width = (max_conf - min_conf)/100
        if self.bin_spacing == -1:
            self.bin_spacing = (max_conf - min_conf)/10

        # define the bins (according to width)
        self.bins = np.arange(min_conf, max_conf + self.bin_width, self.bin_spacing)
        return self.bins

    def bins_num(self):
        """ Give the number of ctr used for the age bin each participant is in.

        Returns
        -------
        array
            Number of controls in each bin.
        """
        if self.bins is None:
            self.create_bins()

        dists = [np.abs(conf - self.bins) for conf in self.data[self.conf]]
        idx = [np.argmin(d) for d in dists]
        n_ctr = [self.bin_count[i] for i in idx]
        self.data['participants'] = n_ctr
        return n_ctr

    def _loess_rank(self):
        """ Associate ranks to LOESS normative scores."""
        self.data.loc[(self.data.LOESS_pred <= -2), 'LOESS_rank'] = -2
        self.data.loc[(self.data.LOESS_pred > -2) &
                      (self.data.LOESS_pred <= -1), 'LOESS_rank'] = -1
        self.data.loc[(self.data.LOESS_pred > -1) &
                      (self.data.LOESS_pred <= +1), 'LOESS_rank'] = 0
        self.data.loc[(self.data.LOESS_pred > +1) &
                      (self.data.LOESS_pred <= +2), 'LOESS_rank'] = 1
        self.data.loc[(self.data.LOESS_pred > +2), 'LOESS_rank'] = 2

    def loess_normative_model(self):
        """ Compute classical normative model."""
        if self.bins is None:
            self._create_bins()
        
        # format data
        data = self.data[[self.conf, self.score]].to_numpy(dtype=np.float64)

        # take the controls
        ctr_mask, _ = self._get_masks()
        ctr = data[ctr_mask]

        self.zm = np.zeros(self.bins.shape[0])  # mean
        self.zstd = np.zeros(self.bins.shape[0])  # standard deviation
        self.zci = np.zeros([self.bins.shape[0], 2])  # confidence interval

        for i, bin_center in enumerate(self.bins):
            mu = np.array(bin_center)  # bin_center value (age or conf)
            bin_mask = (abs(ctr[:, :1] - mu) < self.bin_width) * 1.
            idx = [u for (u, v) in np.argwhere(bin_mask)]

            scores = ctr[idx, 1]
            adj_conf = ctr[idx, 0] - mu  # confound relative to bin center

            # if more than 2 non NaN values do the model
            if (~np.isnan(scores)).sum() > 2:
                mod = sm.WLS(scores, sm.tools.add_constant(adj_conf, 
                                                           has_constant='add'),
                             missing='drop', weight=bin_mask.flatten()[idx],
                             hasconst=True).fit()
                self.zm[i] = mod.params[0]  # mean

                # std and confidence intervals
                prstd, iv_l, iv_u = wls_prediction_std(mod, [0, 0])
                self.zstd[i] = prstd
                self.zci[i, :] = mod.conf_int()[0, :]  # [iv_l, iv_u]

            else:
                self.zm[i] = np.nan
                self.zci[i] = np.nan
                self.zstd[i] = np.nan

        dists = [np.abs(conf - self.bins) for conf in self.data[self.conf]]
        idx = [np.argmin(d) for d in dists]
        m = np.array([self.zm[i] for i in idx])
        std = np.array([self.zstd[i] for i in idx])
        #nmodel = (self.data[self.score] - m) / std
        #self.data['LOESS_pred'] = nmodel
        self.data['LOESS_pred'] = m
        self.data['LOESS_residuals'] = self.data[self.score] - self.data['LOESS_pred']

        score = self._get_score()
        res = self.data['LOESS_residuals'].to_numpy(dtype=np.float64)
        self.SMSE_LOESS = (np.mean(res[ctr_mask]**2)**0.5) / np.std(score[ctr_mask])

        self._loess_rank()

    def _centiles_rank(self):
        """ Associate ranks to centiles associated with normative modeling."""
        self.data.loc[(self.data.Centiles_pred <= 5), 'Centiles_rank'] = -2
        self.data.loc[(self.data.Centiles_pred > 5) &
                      (self.data.Centiles_pred <= 25), 'Centiles_rank'] = -1
        self.data.loc[(self.data.Centiles_pred > 25) &
                      (self.data.Centiles_pred <= 75), 'Centiles_rank'] = 0
        self.data.loc[(self.data.Centiles_pred > 75) &
                      (self.data.Centiles_pred <= 95), 'Centiles_rank'] = 1
        self.data.loc[(self.data.Centiles_pred > 95), 'Centiles_rank'] = 2

    def centiles_normative_model(self):
        """ Compute centiles normative model."""
        if self.bins is None:
            self._create_bins()

        # format data
        data = self.data[[self.conf, self.score]].to_numpy(dtype=np.float64)

        # take the controls
        ctr_mask, _ = self._get_masks()
        ctr = data[ctr_mask]

        self.z = np.zeros([self.bins.shape[0], 101])  # centiles

        for i, bin_center in enumerate(self.bins):
            mu = np.array(bin_center)  # bin_center value (age or conf)
            bin_mask = (abs(ctr[:, :1] - mu) <
                        self.bin_width) * 1.  # one hot mask
            idx = [u for (u, v) in np.argwhere(bin_mask)]
            scores = ctr[idx, 1]

            # if more than 2 non NaN values do the model
            if (~np.isnan(scores)).sum() > 2:
                # centiles
                self.z[i, :] = mquantiles(scores, prob=np.linspace(0, 1, 101), alphap=0.4, betap=0.4)
            else:
                self.z[i] = np.nan

        dists = [np.abs(conf - self.bins) for conf in self.data[self.conf]]
        idx = [np.argmin(d) for d in dists]
        centiles = np.array([self.z[i] for i in idx])
        result = np.zeros(centiles.shape[0])
        max_mask = self.data[self.score] >= np.max(centiles, axis=1)
        min_mask = self.data[self.score] < np.min(centiles, axis=1)
        else_mask = ~(max_mask | min_mask)
        result[max_mask] = 100
        result[min_mask] = 0
        result[else_mask] = np.array([np.argmin(self.data[self.score][i] >= centiles[i]) for i in range(self.data.shape[0])])[else_mask]
        self.data['Centiles'] = result
        self.data['Centiles_pred'] = np.array([centiles[i, 50] for i in range(self.data.shape[0])])
        self.data['Centiles_residuals'] = self.data[self.score] - self.data['Centiles_pred']

        score = self._get_score()
        res = self.data['Centiles_residuals'].to_numpy(dtype=np.float64)
        self.SMSE_Centiles = (np.mean(res[ctr_mask]**2)**0.5) / np.std(score[ctr_mask])

        self._centiles_rank()

    def _get_conf_mat(self):
        """ Get confounds properly formatted from dataframe and input list.

        Returns
        -------
        array
            Confounds with categorical values dummy encoded. Dummy encoding keeps k-1
            dummies out of k categorical levels.
        """
        conf_clean, conf_cat = _read_confounds(self.confounds)
        conf_mat = pd.get_dummies(self.data[conf_clean], columns=conf_cat, 
                                  drop_first=True)
        return conf_mat.to_numpy()

    def _get_score(self):
        """ Get the score from the PyNM object as an array.

        Raises
        ------
        ValueError
            Method must be one of "auto","approx", or "exact".

        Returns
        -------
        array
            The column of data marked by the user as 'score'.
        """
        return self.data[self.score].to_numpy()

    def _use_approx(self, method='auto'):
        """ Choose wether or not to use SVGP model. If method is set to 'auto' SVGP is chosen
        for datasets with more than 2000 points.

        Parameters
        ----------
        method: str, default='auto'
            Which method to use, can be 'exact' for exact GP regression, 'approx' for SVGP,
            or 'auto' which will set the method according to the size of the data.
        
        Raises
        ------
        ValueError
            Method must be one of "auto","approx", or "exact".
        """
        if method == 'auto':
            if self.data.shape[0] > 2000:
                return True
            else:
                return False
        elif method == 'approx':
            return True
        elif method == 'exact':
            if self.data.shape[0] > 2000:
                warnings.warn("Exact GP model with over 2000 data points requires "
                              "large amounts of time and memory, continuing with exact model.",Warning)
            return False
        else:
            raise ValueError('Method must be one of "auto","approx", or "exact".')
    
    def _test_gp_residuals(self,conf_mat):
        #Test normal
        k2, p_norm = stats.normaltest(self.data['GP_residuals'])
        if p_norm < 0.05:
            warnings.warn("The residuals are not Gaussian!")
        
        # Test heteroskedasticity
        exog = add_constant(conf_mat)
        _,p_het,_,_ = het_white((self.data['GP_residuals'])**2,exog)
        if p_het < 0.05:
            warnings.warn("The residuals are heteroskedastic!")
        
    def gp_normative_model(self, length_scale=1, nu=2.5, method='auto', batch_size=256, n_inducing=500, num_epochs=20):
        """ Compute gaussian process normative model. Gaussian process regression is computed using
        the Matern Kernel with an added constant and white noise. For Matern kernel see scikit-learn documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html.

        Parameters
        -------
        length_scale: float, default=1
            Length scale parameter of Matern kernel.
        nu: float, default=2.5
            Nu parameter of Matern kernel.
        method: str, default='auto'
            Which method to use, can be 'exact' for exact GP regression, 'approx' for SVGP,
            or 'auto' which will set the method according to the size of the data.
        batch_size: int, default=256
            Batch size for SVGP model training and prediction.
        n_inducing: int, default=500
            Number of inducing points for SVGP model.
        num_epochs: int, default=20
            Number of epochs (passes through entire dataset) to train SVGP for.
        """
        # get proband and control masks
        ctr_mask, _ = self._get_masks()

        # get matrix of confounds
        conf_mat = self._get_conf_mat()

        # Define independent and response variables
        y = self.data[self.score][ctr_mask].to_numpy().reshape(-1, 1)
        X = conf_mat[ctr_mask]
        
        score = self._get_score()
        
        if self._use_approx(method=method):
            self.loss = self._svgp_normative_model(conf_mat,score,ctr_mask,nu=nu,batch_size=batch_size,n_inducing=n_inducing,num_epochs=num_epochs)

        else:
            # Define independent and response variables
            y = score[ctr_mask].reshape(-1,1)
            X = conf_mat[ctr_mask]

            # Fit normative model on controls
            kernel = ConstantKernel() + WhiteKernel(noise_level=1) + Matern(length_scale=length_scale, nu=nu)
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
            gp.fit(X, y)

            #Predict normative values
            y_pred, sigma = gp.predict(conf_mat, return_std=True)
            y_true = self.data[self.score].to_numpy().reshape(-1,1)
            residuals = y_true - y_pred
            self.SMSE_GP = (np.mean((residuals)**2))**0.5 / \
                np.std(score[ctr_mask])

            SLL = ( 0.5 * np.log(2 * np.pi * sigma**2) +
                   (residuals)**2 / (2 * sigma**2) -
                   (y_true - np.mean(score[ctr_mask]))**2 /
                   (2 * np.std(score[ctr_mask])) )

            self.MSLL = np.mean(SLL)

            self.data['GP_pred'] = y_pred
            self.data['GP_sigma'] = sigma
            self.data['GP_residuals'] = residuals

        self._test_gp_residuals(conf_mat)

    def _svgp_normative_model(self,conf_mat,score,ctr_mask,nu=2.5,batch_size=256,n_inducing=500,num_epochs=20):
        """ Compute SVGP model. See GPyTorch documentation for further details:
        https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html#Creating-a-SVGP-Model.

        Parameters
        ----------
        conf_mat: array
            Confounds with categorical values dummy encoded.
        score: array
            Score/response variable.
        ctr_mask: array
            Mask (boolean array) with controls marked True.
        nu: float, default=2.5
            Nu parameter of Matern kernel.
        batch_size: int, default=256
            Batch size for SVGP model training and prediction.
        n_inducing: int, default=500
            Number of inducing points for SVGP model.
        num_epochs: int, default=20
            Number of epochs (passes through entire dataset) to train SVGP for.
        
        Raises
        ------
        ImportError
            GPyTorch or it's dependencies aren't installed.

        Returns
        -------
        array
            Loss per epoch of training (temp for debugging).
        """
        try:
            from pynm.approx import SVGP
        except:
            raise ImportError("GPyTorch and it's dependencies must be installed to use the SVGP model.")
        else:
            svgp = SVGP(conf_mat,score,ctr_mask,n_inducing=n_inducing,batch_size=batch_size)
            svgp.train(num_epochs=num_epochs)
            means, sigma = svgp.predict()

            y_pred = means.numpy()
            y_true = score
            residuals = (y_true - y_pred).astype(float)

            self.SMSE_GP = (np.mean(y_true - y_pred)**2)**0.5 / np.std(score[ctr_mask])

            SLL = (0.5 * np.log(2 * np.pi * sigma.numpy()**2) +
                   (y_true - y_pred)**2 / (2 * sigma.numpy()**2) -
                   (y_true - np.mean(score[ctr_mask]))**2 /
                   (2 * np.std(score[ctr_mask])) )

            self.MSLL = np.mean(SLL)

            self.data['GP_pred'] = y_pred
            self.data['GP_sigma'] = sigma.numpy()
            self.data['GP_residuals'] = residuals
            return svgp.loss
    
    def gamlss_normative_model(self,mu=None,sigma=None,nu=None,tau=None,family='SHASHo2',what='mu',lib_loc=None):
        """Compute GAMLSS normative model.
        
        Parameters
        ----------
        mu: str or None
            Formula for mu (location) parameter of GAMLSS. If None, formula for score is sum of confounds
            with non-categorical columns as smooth functions, e.g. "score ~ ps(age) + sex".
        sigma: str or None
            Formula for mu (location) parameter of GAMLSS. If None, formula is '~ 1'.
        nu: str or None
            Formula for mu (location) parameter of GAMLSS. If None, formula is '~ 1'.
        tau: str or None
            Formula for mu (location) parameter of GAMLSS. If None, formula is '~ 1'.
        family: str,default='SHASHo2'
            Family of distributions to use for fitting, default is 'SHASHo2'. See R documentation for GAMLSS package for other available families of distributions.
        what: str, default='mu'
            What parameter to predict, can be 'mu', 'sigma', 'nu' or 'tau'.
        lib_loc: str, default=None
            Path to location of installed GAMLSS package.
        
        Notes
        -----
        If using 'random()' to model a random effect in any of the formulas, it must be passed a column of the dataframe with categorical values
        as a factor: e.g. 'random(as.factor(COL))'. Using a random effect also impacts which parameter it is possible to predict i.e. set the 'what'
        argument accordingly.
        """
        try:
            from pynm.gamlss import GAMLSS
        except:
            raise ImportError("R and the GAMLSS package must be installed to use GAMLSS model, see documentation for installation help.")
        else:
            # get proband and control masks
            ctr_mask, _ = self._get_masks()

            gamlss = GAMLSS(mu=mu,sigma=sigma,nu=nu,tau=tau,family=family,what=what,lib_loc=lib_loc,score=self.score,confounds=self.confounds)
            #TODO: more graceful fix for this problem
            nan_cols = ['LOESS_pred','LOESS_residuals','LOESS_rank','Centiles_pred','Centiles_residuals','Centiles','Centiles_rank']
            gamlss_data = self.data[[c for c in self.data.columns if c not in nan_cols]]
            gamlss.fit(gamlss_data[ctr_mask])
            res = gamlss.predict(gamlss_data)

            self.data['GAMLSS_pred'] = res
            self.data['GAMLSS_residuals'] = self.data[self.score] - self.data['GAMLSS_pred']

            #TODO: test residuals?

    def _plot(self, ax,kind=None,gp_xaxis=None,gamlss_xaxis=None):
        """ Plot the data with the normative model overlaid.

        Parameters
        ----------
        ax: matplotlib axis
            Axis on which to plot.
        kind: str, default=None
            Type of plot among "LOESS" (local polynomial), "Centiles", "GP" (gaussian processes), 
            or "GAMLSS" (generalized additive models of location scale and shape). 
        gp_xaxis: str,default=None
            Which confound to use for xaxis of GP plot. If set to None, first confound in list passed to model will be used.
        gamlss_xaxis: str,default=None
            Which confound to use for xaxis of GAMLSS plot. If set to None, first confound in list passed to model will be used.

        Returns
        -------
        Axis
            handle for the matplotlib axis of the plot
        """
        if kind is None:
            sns.scatterplot(data=self.data, x=self.conf, y=self.score,
                             hue=self.group, style=self.group,ax=ax)
        elif kind == 'LOESS':
            sns.scatterplot(data=self.data, x=self.conf, y=self.score,
                             hue=self.group, style=self.group,ax=ax)
            tmp=self.data.sort_values(self.conf)
            #ax.plot(self.bins, self.zm, '-k')
            ax.plot(tmp[self.conf], tmp['LOESS_pred'], '-k')
            #plt.fill_between(np.squeeze(self.bins),
            #                 np.squeeze(self.zm) - 2 * np.squeeze(self.zstd),
            #                 np.squeeze(self.zm) + 2 * np.squeeze(self.zstd),
            #                 alpha=.2, fc='grey', ec='None', label='95% CI')
        elif kind == 'Centiles':
            sns.scatterplot(data=self.data, x=self.conf, y=self.score,
                                hue=self.group, style=self.group,ax=ax)
            tmp=self.data.sort_values(self.conf)
            #ax.plot(self.bins, self.z[:, 50], '--k')
            ax.plot(tmp[self.conf], tmp['LOESS_pred'], '-k')
            #plt.fill_between(np.squeeze(self.bins),
            #                 np.squeeze(self.z[:, 5]),
            #                 np.squeeze(self.z[:, 95]),
            #                 alpha=.2, fc='grey', ec='None', label='95% CI')
        elif kind == 'GP':
            if gp_xaxis is None:
                gp_xaxis = self.conf
            sns.scatterplot(data=self.data, x=gp_xaxis, y=self.score,
                                hue=self.group, style=self.group,ax=ax)
            tmp=self.data.sort_values(gp_xaxis)
            #plt.fill_between(np.squeeze(tmp[gp_xaxis]),
            #                 np.squeeze(tmp['GP_pred']) - 2*np.squeeze(tmp['GP_sigma']),
            #                 np.squeeze(tmp['GP_pred']) + 2*np.squeeze(tmp['GP_sigma']),
            #                 alpha=.2, fc='grey', ec='None', label='95% CI')
            ax.plot(tmp[gp_xaxis], tmp['GP_pred'], '-k')
        elif kind == 'GAMLSS':
            if gamlss_xaxis is None:
                gamlss_xaxis = self.conf
            sns.scatterplot(data=self.data, x=gamlss_xaxis, y=self.score,
                                hue=self.group, style=self.group,ax=ax)
            tmp=self.data.sort_values(gamlss_xaxis)
            #plt.fill_between(np.squeeze(tmp[gp_xaxis]),
            #                 np.squeeze(tmp['GP_pred']) - 2*np.squeeze(tmp['GP_sigma']),
            #                 np.squeeze(tmp['GP_pred']) + 2*np.squeeze(tmp['GP_sigma']),
            #                 alpha=.2, fc='grey', ec='None', label='95% CI')
            ax.plot(tmp[gamlss_xaxis], tmp['GAMLSS_pred'], '-k')
        return ax

    def plot(self, kind=None,gp_xaxis=None,gamlss_xaxis=None):
        """Plot the data with the normative model overlaid.

        Parameters
        ----------
        kind: list, default=None
            Type of plot, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None. If None, all available
            results will be plotted, if None are available a warning will be raised and only the data will be plotted.
        gp_xaxis: str,default=None
            Which confound to use for xaxis of GP plot. If set to None, first confound in list passed to model will be used.
        gamlss_xaxis: str,default=None
            Which confound to use for xaxis of GAMLSS plot. If set to None, first confound in list passed to model will be used.
        
        Raises
        ------
        ValueError
            Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.
        """
        if kind is None:
            kind = []
            for k in ['LOESS','Centiles','GP','GAMLSS']:
                if '{}_pred'.format(k) in self.data.columns:
                    kind.append(k)
            if len(kind)==0:
                warnings.warn('No model results found in data.')
        
        if set(kind).issubset(set(['LOESS','Centiles','GP','GAMLSS'])) and len(kind)>0:
            fig, ax = plt.subplots(1,len(kind),figsize=(len(kind)*5,5))
            for i,k in enumerate(kind):
                self._plot(ax[i],kind=k,gp_xaxis=gp_xaxis,gamlss_xaxis=gamlss_xaxis)
                ax[i].set_title(k)
            plt.show()
        elif len(kind)==0:
            fig, ax = plt.subplots(1,1)
            self._plot(ax,None,gp_xaxis=gp_xaxis,gamlss_xaxis=gamlss_xaxis)
            ax.set_title('Data')
            plt.show()
        else:
            raise ValueError('Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.')

    def _plot_res(self, ax,kind=None, confound=None):
        """ Plot the residuals of the normative model.

        Parameters
        ----------
        ax: matplotlib axis
            Axis on which to plot.
        kind: str, default=None
            Type of plot among "LOESS" (local polynomial), "Centiles", "GP" (gaussian processes),
            or "GAMLSS" (generalized additive models of location scale and shape).
        confound: str or None
            Which confound to use as xaxis of plot, must be categorical or None.
        """
        if confound is None:
            confound = np.zeros(self.data.shape[0])
        if kind == 'LOESS':
            sns.violinplot(x=confound, y='LOESS_residuals',
                           data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_LOESS,3)}")
        if kind == 'Centiles':
            sns.violinplot(x=confound, y='Centiles_residuals',
                           data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_Centiles,3)}")
        if kind == 'GP':
            sns.violinplot(x=confound, y='GP_residuals',
                           data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_GP,3)} - MSLL={np.round(self.MSLL,3)}")
        if kind == 'GAMLSS':
            sns.violinplot(x=confound, y='GAMLSS_residuals',
                           data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            #ax.set_title(f"{kind} SMSE={np.round(self.SMSE_GAMLSS,3)} - MSLL={np.round(self.MSLL,3)}")
            ax.set_title(f"{kind}")
        if not isinstance(confound,str):
            ax.set_xticklabels([''])

    def plot_res(self, kind=None, confound=None):
        """Plot the residuals of the normative model.

        Parameters
        ----------
        kind: list default=None
            Type of plot, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None. If None, all available
            results will be plotted, if None are available a ValueError will be raised.
        confound: str, default=None
            Which confound to use as xaxis of plot, must be categorical or None.
        
        Raises
        ------
        ValueError
            Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.
        ValueError
            No model results found in data.
        """
        if kind is None:
            kind = []
            for k in ['LOESS','Centiles','GP','GAMLSS']:
                if '{}_pred'.format(k) in self.data.columns:
                    kind.append(k)
            if len(kind)==0:
                raise ValueError('No model results found in data.')
        
        if set(kind).issubset(set(['LOESS','Centiles','GP','GAMLSS'])):
            fig, ax = plt.subplots(1,len(kind),figsize=(len(kind)*5,5))
            for i,k in enumerate(kind):
                self._plot_res(ax[i],kind=k,confound=confound)
            plt.show()
        else:
            raise ValueError('Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.')
