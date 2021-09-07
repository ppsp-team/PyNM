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
from pynm.util import * 

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
    RMSE_LOESS: float
        RMSE of LOESS normative model
    SMSE_LOESS: float
        SMSE of LOESS normative model
    RMSE_Centiles: float
        RMSE of Centiles normative model
    SMSE_Centiles: float
        SMSE of Centiles normative model
    RMSE_GP: float
        RMSE of Gaussian Process normative model
    SMSE_GP: float
        SMSE of Gaussian Process normative model
    MSLL_GP: float
        MSLL of Gaussian Process normative model
    RMSE_GAMLSS: float
        RMSE of GAMLSS
    SMSE_GAMLSS: float
        SMSE of GAMLSS
    MSLL_GAMLSS: float
        MSLL of GAMLSS
    """

    def __init__(self, data, score, group, confounds, 
                train_sample='controls', bin_spacing=-1, bin_width=-1):
        """ Create a PyNM object.

        Parameters
        ----------
        data : dataframe
            Dataset to fit model, must at least contain columns corresponding to 'group',
            'score', and 'conf'.
        score : str
            Label of column from data with score (response variable).
        group : str
            Label of column from data that encodes wether subjects are probands or controls.
        confounds: list of str
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
        
        Raises
        ------
        ValueError
            Each row of DataFrame must have a unique index.
        """
        if data.index.nunique() != data.shape[0]:
            raise ValueError('Each row of DataFrame must have a unique index.')
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
        self.RMSE_LOESS = None
        self.SMSE_LOESS = None
        self.RMSE_Centiles = None
        self.SMSE_Centiles = None
        self.RMSE_GP = None
        self.SMSE_GP = None
        self.MSLL_GP = None
        self.RMSE_GAMLSS = None
        self.SMSE_GAMLSS = None
        self.MSLL_GAMLSS = None

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
        self.data.loc[(self.data.LOESS_z <= -2), 'LOESS_rank'] = -2
        self.data.loc[(self.data.LOESS_z > -2) &
                      (self.data.LOESS_z <= -1), 'LOESS_rank'] = -1
        self.data.loc[(self.data.LOESS_z > -1) &
                      (self.data.LOESS_z <= +1), 'LOESS_rank'] = 0
        self.data.loc[(self.data.LOESS_z > +1) &
                      (self.data.LOESS_z <= +2), 'LOESS_rank'] = 1
        self.data.loc[(self.data.LOESS_z > +2), 'LOESS_rank'] = 2

    def loess_normative_model(self):
        """ Compute LOESS normative model."""
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

        self.data['LOESS_pred'] = m
        self.data['LOESS_sigma'] = std
        self.data['LOESS_residuals'] = self.data[self.score] - self.data['LOESS_pred']
        self.data['LOESS_z'] = self.data['LOESS_residuals']/self.data['LOESS_sigma']

        self.RMSE_LOESS = RMSE(self.data[self.score].values[ctr_mask],m[ctr_mask])
        self.SMSE_LOESS = SMSE(self.data[self.score].values[ctr_mask],m[ctr_mask])

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
        centiles_50 = np.array([centiles[i, 50] for i in range(self.data.shape[0])])
        centiles_68 = np.array([centiles[i, 68] for i in range(self.data.shape[0])])
        centiles_32 = np.array([centiles[i, 32] for i in range(self.data.shape[0])])

        result = np.zeros(centiles.shape[0])
        max_mask = self.data[self.score] >= np.max(centiles, axis=1)
        min_mask = self.data[self.score] < np.min(centiles, axis=1)
        else_mask = ~(max_mask | min_mask)
        result[max_mask] = 100
        result[min_mask] = 0
        result[else_mask] = np.array([np.argmin(self.data[self.score][i] >= centiles[i]) for i in range(self.data.shape[0])])[else_mask]

        self.data['Centiles'] = result
        self.data['Centiles_pred'] = centiles_50
        # TODO: Correct Centiles_sigma
        # avg of 68 - 50th percentile and 50-32th percentile (assume normal) (DOESN"T MAKE SENSE PLOTTED)
        self.data['Centiles_95'] = np.array([centiles[i, 95] for i in range(self.data.shape[0])])
        self.data['Centiles_5'] = np.array([centiles[i, 5] for i in range(self.data.shape[0])])
        self.data['Centiles_sigma'] = (centiles_68 - centiles_32)/2
        self.data['Centiles_residuals'] = self.data[self.score] - self.data['Centiles_pred']
        self.data['Centiles_z'] = self.data['Centiles_residuals']/self.data['Centiles_sigma']

        self.RMSE_Centiles = RMSE(self.data[self.score].values[ctr_mask],self.data['Centiles_pred'].values[ctr_mask])
        self.SMSE_Centiles = SMSE(self.data[self.score].values[ctr_mask],self.data['Centiles_pred'].values[ctr_mask])

        self._centiles_rank()

    def _get_conf_mat(self):
        """ Get confounds properly formatted from dataframe and input list.

        Returns
        -------
        array
            Confounds with categorical values dummy encoded. Dummy encoding keeps k-1
            dummies out of k categorical levels.
        """
        conf_clean, conf_cat = read_confounds(self.confounds)
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
        
    def gp_normative_model(self, length_scale=1, nu=2.5, length_scale_bounds=(1e-5,1e5),method='auto', batch_size=256, n_inducing=500, num_epochs=20):
        """ Compute gaussian process normative model. Gaussian process regression is computed using
        the Matern Kernel with an added constant and white noise. For Matern kernel see scikit-learn documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html.

        Parameters
        -------
        length_scale: float, default=1
            Length scale parameter of Matern kernel.
        nu: float, default=2.5
            Nu parameter of Matern kernel.
        length_scale_bounds: pair of floats >= 0 or 'fixed', default=(1e-5, 1e5)
            The lower and upper bound on length_scale. If set to 'fixed', ‘length_scale’ cannot be changed during hyperparameter tuning.
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
            self.loss = self._svgp_normative_model(conf_mat,score,ctr_mask,nu=nu,length_scale=length_scale, length_scale_bounds=length_scale_bounds,
                                                batch_size=batch_size,n_inducing=n_inducing,num_epochs=num_epochs)

        else:
            # Define independent and response variables
            y = score[ctr_mask].reshape(-1,1)
            X = conf_mat[ctr_mask]

            # Fit normative model on controls
            kernel = ConstantKernel() + WhiteKernel(noise_level=1) + Matern(length_scale=length_scale, nu=nu,length_scale_bounds=length_scale_bounds)
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
            gp.fit(X, y)

            #Predict normative values
            y_pred, sigma = gp.predict(conf_mat, return_std=True)
            y_true = self.data[self.score].to_numpy().reshape(-1,1)
            residuals = y_true - y_pred

            self.data['GP_pred'] = y_pred
            self.data['GP_sigma'] = sigma
            self.data['GP_residuals'] = residuals
            self.data['GP_z'] = self.data['GP_residuals'] / self.data['GP_sigma']

            self.RMSE_GP = RMSE(y_true[ctr_mask],y_pred[ctr_mask])
            self.SMSE_GP = SMSE(y_true[ctr_mask],y_pred[ctr_mask])
            self.MSLL_GP = MSLL(y_true[ctr_mask],y_pred[ctr_mask],sigma[ctr_mask])

        self._test_gp_residuals(conf_mat)

    def _svgp_normative_model(self,conf_mat,score,ctr_mask,nu=2.5,length_scale=1,length_scale_bounds=(1e-5,1e5),batch_size=256,n_inducing=500,num_epochs=20):
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
        length_scale: float, default=1
            Length scale parameter of Matern kernel.
        length_scale_bounds: pair of floats >= 0 or 'fixed', default=(1e-5, 1e5)
            The lower and upper bound on length_scale. If set to 'fixed', ‘length_scale’ cannot be changed during hyperparameter tuning.
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
            svgp = SVGP(conf_mat,score,ctr_mask,n_inducing=n_inducing,batch_size=batch_size,nu=nu,length_scale=length_scale,length_scale_bounds=length_scale_bounds)
            svgp.train(num_epochs=num_epochs)
            means, sigma = svgp.predict()

            y_pred = means.numpy()
            y_true = score
            residuals = (y_true - y_pred).astype(float)

            self.data['GP_pred'] = y_pred
            self.data['GP_sigma'] = sigma.numpy()
            self.data['GP_residuals'] = residuals
            self.data['GP_z'] = self.data['GP_residuals']/self.data['GP_sigma']

            self.RMSE_GP = RMSE(y_true[ctr_mask],y_pred[ctr_mask])
            self.SMSE_GP = SMSE(y_true[ctr_mask],y_pred[ctr_mask])
            self.MSLL_GP = MSLL(y_true[ctr_mask],y_pred[ctr_mask],sigma.numpy()[ctr_mask])

    
    def gamlss_normative_model(self,mu=None,sigma=None,nu=None,tau=None,family='SHASHo2',method='RS',lib_loc=None):
        """Compute GAMLSS normative model.
        
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
        family: str,default='SHASHo2'
            Family of distributions to use for fitting, default is 'SHASHo2'. See R documentation for GAMLSS package for other available families of distributions.
        method: str, default = 'RS'
            Method for fitting GAMLSS. Can be 'RS' (Rigby and Stasinopoulos algorithm), 'CG' (Cole and Green algorithm) or 'mixed(n,m)' where n & m are integers.
            Specifying 'mixed(n,m)' will use the RS algorithm for n iterations and the CG algorithm for up to m additional iterations.
        lib_loc: str, default=None
            Path to location of installed GAMLSS package.
        
        Notes
        -----
        If using 'random()' to model a random effect in any of the formulas, it must be passed a column of the dataframe with categorical values
        as a factor: e.g. 'random(as.factor(COL))'.
        """
        try:
            from pynm.gamlss import GAMLSS
        except:
            raise ImportError("R and the GAMLSS package must be installed to use GAMLSS model, see documentation for installation help.")
        else:
            # get proband and control masks
            ctr_mask, _ = self._get_masks()

            gamlss = GAMLSS(mu=mu,sigma=sigma,nu=nu,tau=tau,family=family,method=method,
                            lib_loc=lib_loc,score=self.score,confounds=self.confounds)

            nan_cols = ['LOESS_pred','LOESS_residuals','LOESS_z','LOESS_rank','LOESS_sigma',
            'Centiles_pred','Centiles_residuals','Centiles_z','Centiles','Centiles_rank','Centiles_sigma',
            'Centiles_95','Centiles_5']
            gamlss_data = self.data[[c for c in self.data.columns if c not in nan_cols]]

            gamlss.fit(gamlss_data[ctr_mask])
            
            try:
                mu_pred = gamlss.predict(gamlss_data,what='mu')
                sigma_pred = gamlss.predict(gamlss_data,what='sigma')
            except:
                raise RuntimeError("GAMLSS fitting algorithm has not yet converged - adjust 'method' parameter.")
            else:
                self.data['GAMLSS_pred'] = mu_pred
                self.data['GAMLSS_sigma'] = sigma_pred
                self.data['GAMLSS_residuals'] = self.data[self.score] - self.data['GAMLSS_pred']
                self.data['GAMLSS_z'] = self.data['GAMLSS_residuals']/self.data['GAMLSS_sigma']

                self.RMSE_GAMLSS = RMSE(mu_pred[ctr_mask],self.data[self.score].values[ctr_mask])
                self.SMSE_GAMLSS = SMSE(mu_pred[ctr_mask],self.data[self.score].values[ctr_mask])
                self.MSLL_GAMLSS = MSLL(mu_pred[ctr_mask],self.data[self.score].values[ctr_mask],sigma_pred[ctr_mask])

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
            ax.plot(tmp[self.conf], tmp['LOESS_pred'], '-k',label='Prediction')
            ax.plot(tmp[self.conf], tmp['LOESS_pred'] - 1.96*tmp['LOESS_sigma'], '--k')
            ax.plot(tmp[self.conf], tmp['LOESS_pred'] + 1.96*tmp['LOESS_sigma'], '--k',label='95% CI')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
        elif kind == 'Centiles':
            sns.scatterplot(data=self.data, x=self.conf, y=self.score,
                                hue=self.group, style=self.group,ax=ax)
            tmp=self.data.sort_values(self.conf)
            ax.plot(tmp[self.conf], tmp['Centiles_pred'], '-k',label='Prediction')
            ax.plot(tmp[self.conf], tmp['Centiles_5'],'--k')
            ax.plot(tmp[self.conf], tmp['Centiles_95'],'--k',label='95% CI')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
        elif kind == 'GP':
            if gp_xaxis is None:
                gp_xaxis = self.conf
            sns.scatterplot(data=self.data, x=gp_xaxis, y=self.score,
                                hue=self.group, style=self.group,ax=ax)
            tmp=self.data.sort_values(gp_xaxis)
            if len(self.confounds) == 1:
                ax.plot(tmp[gp_xaxis], tmp['GP_pred'], '-k',label='Prediction')
                ax.plot(tmp[gp_xaxis], tmp['GP_pred'] - 1.96*tmp['GP_sigma'], '--k')
                ax.plot(tmp[gp_xaxis], tmp['GP_pred'] + 1.96*tmp['GP_sigma'], '--k',label='95% CI')
            else:
                ax.scatter(tmp[gp_xaxis], tmp['GP_pred'], label='Prediction',color='black',marker='_',s=25)
                ax.scatter(tmp[gp_xaxis], tmp['GP_pred'] - 1.96*tmp['GP_sigma'],color='black',s=0.2)
                ax.scatter(tmp[gp_xaxis], tmp['GP_pred'] + 1.96*tmp['GP_sigma'], label='95% CI',color='black',s=0.2)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
        elif kind == 'GAMLSS':
            if gamlss_xaxis is None:
                gamlss_xaxis = self.conf
            sns.scatterplot(data=self.data, x=gamlss_xaxis, y=self.score,
                                hue=self.group, style=self.group,ax=ax)
            tmp=self.data.sort_values(gamlss_xaxis)
            if len(self.confounds) == 1:
                ax.plot(tmp[gamlss_xaxis], tmp['GAMLSS_pred'], '-k',label='Prediction')
                ax.plot(tmp[gamlss_xaxis], tmp['GAMLSS_pred'] - 1.96*tmp['GAMLSS_sigma'], '--k')
                ax.plot(tmp[gamlss_xaxis], tmp['GAMLSS_pred'] + 1.96*tmp['GAMLSS_sigma'], '--k',label='95% CI')
            else:
                ax.scatter(tmp[gamlss_xaxis], tmp['GAMLSS_pred'], label='Prediction',color='black',marker='_',s=25)
                ax.scatter(tmp[gamlss_xaxis], tmp['GAMLSS_pred'] - 1.96*tmp['GAMLSS_sigma'],color='black',s=0.2)
                ax.scatter(tmp[gamlss_xaxis], tmp['GAMLSS_pred'] + 1.96*tmp['GAMLSS_sigma'], label='95% CI',color='black',s=0.2)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
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
        
        if set(kind).issubset(set(['LOESS','Centiles','GP','GAMLSS'])) and len(kind)>1:
            fig, ax = plt.subplots(1,len(kind),figsize=(len(kind)*5,5))
            for i,k in enumerate(kind):
                self._plot(ax[i],kind=k,gp_xaxis=gp_xaxis,gamlss_xaxis=gamlss_xaxis)
                ax[i].set_title(k)
            plt.show()
        elif set(kind).issubset(set(['LOESS','Centiles','GP','GAMLSS'])) and len(kind)>0:
            fig, ax = plt.subplots(1,len(kind),figsize=(len(kind)*5,5))
            self._plot(ax,kind=kind[0],gp_xaxis=gp_xaxis,gamlss_xaxis=gamlss_xaxis)
            ax.set_title(kind[0])
            plt.show()
        elif len(kind)==0:
            fig, ax = plt.subplots(1,1)
            self._plot(ax,None,gp_xaxis=gp_xaxis,gamlss_xaxis=gamlss_xaxis)
            ax.set_title('Data')
            plt.show()
        else:
            raise ValueError('Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.')

    def _plot_res_z(self, ax,kind=None, confound=None,z=False):
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
            if z:
                sns.violinplot(x=confound, y='LOESS_z',
                            data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            else:
                sns.violinplot(x=confound, y='LOESS_residuals',
                            data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_LOESS,3)}")
        if kind == 'Centiles':
            if z:
                sns.violinplot(x=confound, y='Centiles_z',
                           data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            else:
                sns.violinplot(x=confound, y='Centiles_residuals',
                            data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_Centiles,3)}")
        if kind == 'GP':
            if z:
                    sns.violinplot(x=confound, y='GP_z',
                            data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            else:
                sns.violinplot(x=confound, y='GP_residuals',
                            data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_GP,3)} - MSLL={np.round(self.MSLL_GP,3)}")
        if kind == 'GAMLSS':
            if z:
                sns.violinplot(x=confound, y='GAMLSS_z',
                            data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            else:
                sns.violinplot(x=confound, y='GAMLSS_residuals',
                            data=self.data, split=True, palette='Blues', hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_GAMLSS,3)}")
        if not isinstance(confound,str):
            ax.set_xticklabels([''])
    
    def _plot_res_z_cont(self, ax,kind=None, confound=None,z=False):
        """ Plot the residuals of the normative model.

        Parameters
        ----------
        ax: matplotlib axis
            Axis on which to plot.
        kind: str, default=None
            Type of plot among "LOESS" (local polynomial), "Centiles", "GP" (gaussian processes),
            or "GAMLSS" (generalized additive models of location scale and shape).
        confound: str or None
            Which confound to use as xaxis of plot, must be continuous.
        """
        if kind == 'LOESS':
            if z:
                sns.scatterplot(x=confound, y='LOESS_z',
                                data=self.data, hue=self.group,ax=ax)
            else:
                sns.scatterplot(x=confound, y='LOESS_residuals',
                                data=self.data, hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_LOESS,3)}")
        if kind == 'Centiles':
            if z:
                sns.scatterplot(x=confound, y='Centiles_z',
                            data=self.data, hue=self.group,ax=ax)
            else:
                sns.scatterplot(x=confound, y='Centiles_residuals',
                                data=self.data, hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_Centiles,3)}")
        if kind == 'GP':
            if z:
                sns.scatterplot(x=confound, y='GP_z',
                                data=self.data, hue=self.group,ax=ax)
            else:
                sns.scatterplot(x=confound, y='GP_residuals',
                                data=self.data, hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_GP,3)} - MSLL={np.round(self.MSLL_GP,3)}")
        if kind == 'GAMLSS':
            if z:
                sns.scatterplot(x=confound, y='GAMLSS_z',
                                data=self.data, hue=self.group,ax=ax)
            else:
                sns.scatterplot(x=confound, y='GAMLSS_residuals',
                                data=self.data, hue=self.group,ax=ax)
            ax.set_title(f"{kind} SMSE={np.round(self.SMSE_GAMLSS,3)}")

    def plot_res(self, kind=None, confound=None):
        """Plot the residuals of the normative model.

        Parameters
        ----------
        kind: list default=None
            Type of plot, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None. If None, all available
            results will be plotted, if None are available a ValueError will be raised.
        confound: str, default=None
            Which confound to use as xaxis of plot.
        
        Raises
        ------
        ValueError
            Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.
        ValueError
            No model results found in data.
        """
        _, cat = read_confounds(self.confounds)
        if confound is None: 
            categorical = True
        elif confound in cat: 
            categorical = True
        else: 
            categorical = False

        if kind is None:
            kind = []
            for k in ['LOESS','Centiles','GP','GAMLSS']:
                if '{}_residuals'.format(k) in self.data.columns:
                    kind.append(k)
            if len(kind)==0:
                raise ValueError('No model residuals found in data.')
        
        if set(kind).issubset(set(['LOESS','Centiles','GP','GAMLSS'])) and len(kind)>1:
            fig, ax = plt.subplots(1,len(kind),figsize=(len(kind)*5,5))
            for i,k in enumerate(kind):
                if categorical:
                    self._plot_res_z(ax[i],kind=k,confound=confound)
                else:
                    self._plot_res_z_cont(ax[i],kind=k,confound=confound)
            plt.show()
        elif set(kind).issubset(set(['LOESS','Centiles','GP','GAMLSS'])):
            fig, ax = plt.subplots(1,len(kind),figsize=(len(kind)*5,5))
            if categorical:
                self._plot_res_z(ax,kind=kind[0],confound=confound)
            else:
                self._plot_res_z_cont(ax,kind=kind[0],confound=confound)
            plt.show()
        else:
            raise ValueError('Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.')

    def plot_z(self, kind=None, confound=None):
        """Plot the deviance scores of the normative model.

        Parameters
        ----------
        kind: list default=None
            Type of plot, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None. If None, all available
            results will be plotted, if None are available a ValueError will be raised.
        confound: str, default=None
            Which confound to use as xaxis of plot.
        
        Raises
        ------
        ValueError
            Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.
        ValueError
            No model results found in data.
        """
        _, cat = read_confounds(self.confounds)
        if confound is None: 
            categorical = True
        elif confound in cat: 
            categorical = True
        else: 
            categorical = False

        if kind is None:
            kind = []
            for k in ['LOESS','Centiles','GP','GAMLSS']:
                if '{}_z'.format(k) in self.data.columns:
                    kind.append(k)
            if len(kind)==0:
                raise ValueError('No model deviance scores found in data.')
        
        if set(kind).issubset(set(['LOESS','Centiles','GP','GAMLSS'])) and len(kind)>1:
            fig, ax = plt.subplots(1,len(kind),figsize=(len(kind)*5,5))
            for i,k in enumerate(kind):
                if categorical:
                    self._plot_res_z(ax[i],kind=k,confound=confound,z=True)
                else:
                    self._plot_res_z_cont(ax[i],kind=k,confound=confound,z=True)
            plt.show()
        elif set(kind).issubset(set(['LOESS','Centiles','GP','GAMLSS'])):
            fig, ax = plt.subplots(1,len(kind),figsize=(len(kind)*5,5))
            if categorical:
                self._plot_res_z(ax,kind=kind[0],confound=confound,z=True)
            else:
                self._plot_res_z_cont(ax,kind=kind[0],confound=confound,z=True)
            plt.show()
        else:
            raise ValueError('Plot kind not recognized, must be a valid subset of ["Centiles","LOESS","GP","GAMLSS"] or None.')
