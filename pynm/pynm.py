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

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats.mstats import mquantiles

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


def read_confounds(confounds):
    """ Process input list of confounds.

    Parameters
    ----------
    confounds : list of str
        List of confounds with categorical variables indicated by C(var).

    Returns
    -------
    list
        List of all confounds without wrapper on categorical variables: C(var) -> var.
    list
        List of only categorical confounds without wrapper.
    """
    # Find categorical values in confounds and clean format
    categorical = []
    clean_confounds = []
    for conf in confounds:
        if ((conf[0:2] == 'C(') & (conf[-1] == ')')):
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
        List of labels of columns from data with confounds to use for 
        GP model with categorical values denoted by C(var).
    bins:    
    bin_count:
    zm:
    zstd:
    zci:
    z:
    SMSE_LOESS: Mean Square Error of LOESS normative model
    SMSE_Centiles: Median Square Error of Centiles normative model
    SMSE_GP: Mean Square Error of Gaussian Process normative model
    MSLL: Mean Standardized Log Loss of Gaussian Process normative model
    """

    def __init__(self, data, score='score', group='group', conf='age', confounds=['age', 'C(sex)', 'C(site)']):
        self.data = data.copy()
        self.score = score
        self.group = group
        self.CTR = None
        self.PROB = None
        self.conf = conf
        self.confounds = confounds
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

        self.set_group_names()

    def set_group_names(self):
        """ Read whether subjects in data are labeled CTR/PROB or 0/1 and set labels accordingly."""
        labels = list(self.data[self.group].unique())
        if ('CTR' in labels) or ('PROB' in labels):
            self.CTR = 'CTR'
            self.PROB = 'PROB'
        else:
            self.CTR = 0
            self.PROB = 1

    def get_masks(self):
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
    def create_bins(self, min_age=-1, max_age=-1, min_score=-1, max_score=-1,
                    bin_spacing=8, bin_width=1.5):
        """[summary]

        Args:
            min_age (int, optional): [description]. Defaults to -1.
            max_age (int, optional): [description]. Defaults to -1.
            min_score (int, optional): [description]. Defaults to -1.
            max_score (int, optional): [description]. Defaults to -1.
            bin_spacing (int, optional): [description]. Defaults to 8.
            bin_width (float, optional): [description]. Defaults to 1.5.

        Returns:
            [type]: [description]
        """
        if min_age == -1:
            min_age = self.data[self.conf].min()
        if max_age == -1:
            max_age = self.data[self.conf].max()
        if min_score == -1:
            min_score = self.data[self.score].min()
        if max_score == -1:
            max_score = self.data[self.score].max()

        # if max age is more than 300 assume age is in days not years
        if max_age > 300:
            bin_spacing *= 365
            bin_width *= 365

        # define the bins (according to width by age)
        self.bin_width = bin_width
        self.bins = np.arange(min_age, max_age + bin_width, bin_spacing)

        return self.bins

    def bins_num(self):
        """Give the number of ctr used for the age bin each participant is in.

        Returns:
            [type]: [description]
        """
        if self.bins is None:
            self.create_bins()
        dists = [np.abs(conf - self.bins) for conf in self.data[self.conf]]
        idx = [np.argmin(d) for d in dists]
        n_ctr = [self.bin_count[i] for i in idx]
        self.data['participants'] = n_ctr
        return n_ctr

    def loess_rank(self):
        """Associate ranks to LOESS normative scores.
        """
        self.data.loc[(self.data.LOESS_nmodel <= -2), 'LOESS_rank'] = -2
        self.data.loc[(self.data.LOESS_nmodel > -2) &
                      (self.data.LOESS_nmodel <= -1), 'LOESS_rank'] = -1
        self.data.loc[(self.data.LOESS_nmodel > -1) &
                      (self.data.LOESS_nmodel <= +1), 'LOESS_rank'] = 0
        self.data.loc[(self.data.LOESS_nmodel > +1) &
                      (self.data.LOESS_nmodel <= +2), 'LOESS_rank'] = 1
        self.data.loc[(self.data.LOESS_nmodel > +2), 'LOESS_rank'] = 2

    def loess_normative_model(self):
        """Compute classical normative model.

        Returns:
            [type]: [description]
        """
        if self.bins is None:
            self.create_bins()
        # format data
        data = self.data[[self.conf, self.score]].to_numpy(dtype=np.float64)

        # take the controls
        ctr_mask, _ = self.get_masks()
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

        # mean squared error
        MSE = 0

        # for age and score (cols of sel)
        for i in range(ctr.shape[1]):
            idage = np.argmin(np.abs(ctr[i, 1] - self.bins))
            MSE += (ctr[i, 0] - self.zm[idage])**2
        MSE /= ctr.shape[1]
        MSE = self.error_mea**0.5
        self.SMSE_LOESS = MSE / np.std(ctr[:, 1])

        dists = [np.abs(conf - self.bins) for conf in self.data[self.conf]]
        idx = [np.argmin(d) for d in dists]
        m = np.array([self.zm[i] for i in idx])
        std = np.array([self.zstd[i] for i in idx])
        nmodel = (self.data[self.score] - m) / std
        self.data['LOESS_nmodel'] = nmodel
        self.loess_rank()
        return nmodel

    def centiles_rank(self):
        """Associate ranks to centiles associated with normative modeling.
        """
        self.data.loc[(self.data.Centiles_nmodel <= 5), 'Centiles_rank'] = -2
        self.data.loc[(self.data.Centiles_nmodel > 5) &
                      (self.data.Centiles_nmodel <= 25), 'Centiles_rank'] = -1
        self.data.loc[(self.data.Centiles_nmodel > 25) &
                      (self.data.Centiles_nmodel <= 75), 'Centiles_rank'] = 0
        self.data.loc[(self.data.Centiles_nmodel > 75) &
                      (self.data.Centiles_nmodel <= 95), 'Centiles_rank'] = 1
        self.data.loc[(self.data.Centiles_nmodel > 95), 'Centiles_rank'] = 2

    def centiles_normative_model(self):
        """Compute centiles normative model.

        Returns:
            [type]: [description]
        """
        if self.bins is None:
            self.create_bins()

        # format data
        data = self.data[[self.conf, self.score]].to_numpy(dtype=np.float64)

        # take the controls
        ctr_mask, _ = self.get_masks()
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
                self.z[i, :] = mquantiles(scores, prob=np.linspace(
                    0, 1, 101), alphap=0.4, betap=0.4)
            else:
                self.z[i] = np.nan

        # median squared error
        MSE = 0
        # for age and score (cols of sel)
        for i in range(ctr.shape[1]):
            idage = np.argmin(np.abs(ctr[i, 1] - self.bins))
            MSE += (ctr[i, 0] - self.z[idage, 50])**2
        MSE /= ctr.shape[1]
        self.SMSE_Centiles = MSE**0.5 / np.std(ctr[:, 1])

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
        self.data['Centiles_nmodel'] = result
        self.centiles_rank()
        return result

    def get_conf_mat(self):
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
    
    def get_score(self):
        return self.data[self.score].to_numpy()
    
    def use_approx(self,method='auto'):
        if method == 'auto':
            if self.data.shape[0] > 1000:
                return True
            else:
                return False
        elif method == 'approx':
            return True
        elif method == 'exact':
            if self.data.shape[0] > 1000:
                warnings.warn("Exact GP model with over 1000 data points requires "
                                "large amounts of time and memory, continuing with exact model.",Warning)
            return False
        else:
            raise ValueError('Method must be one of "auto","approx", or "exact".')

    def gp_normative_model(self,length_scale=1,nu=2.5, method='auto',batch_size=256,n_inducing=500,num_epochs=20):
        """ Compute gaussian process normative model. Gaussian process regression is computed using
        the Matern Kernel with an added constant and white noise. For Matern kernel see scikit-learn documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html.

        Parameters
        -------
        length_scale: float
            Length scale parameter of Matern kernel.
        nu: float
            Nu parameter of Matern kernel.
        method: str
            Which method to use, can be 'exact' for exact GP regression, 'approx' for SVGP,
            or 'auto' which will set the method according to the size of the data. Default value is 'auto'.
        batch_size: int
            Batch size for SVGP model training and prediction. Default value is 256.
        n_inducing: int
            Number of inducing points for SVGP model. Default value is 500.
        num_epochs: int
            Number of epochs (passes through entire dataset) to train SVGP for. Default value is 20.

        Returns
        -------
        array
            Residuals of normative model.
        """
        # get proband and control masks
        ctr_mask, prob_mask = self.get_masks()

        # get matrix of confounds
        conf_mat = self.get_conf_mat()

        # Define independent and response variables
        y = self.data[self.score][ctr_mask].to_numpy().reshape(-1, 1)
        X = conf_mat[ctr_mask]
        
        score = self.get_score()
        
        if self.use_approx(method=method):
            self.loss = self.svgp_normative_model(conf_mat,score,ctr_mask,nu=nu,batch_size=batch_size,n_inducing=n_inducing,num_epochs=num_epochs)
        
        else:
            #Define independent and response variables
            y = score[ctr_mask].reshape(-1,1)
            X = conf_mat[ctr_mask]

            #Fit normative model on controls
            kernel = ConstantKernel() + WhiteKernel(noise_level=1) + Matern(length_scale=length_scale, nu=nu)
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
            gp.fit(X, y)

            #Predict normative values
            y_pred, sigma = gp.predict(conf_mat, return_std=True)
            y_true = self.data[self.score].to_numpy().reshape(-1,1)

            self.SMSE_GP = (np.mean(y_true - y_pred)**2)**0.5 / np.std(score[ctr_mask])

            SLL = ( 0.5 * np.log(2 * np.pi * sigma**2) +
                     (ytrue - ypred)**2 / (2 * sigma**2) -
                     (ytrue - np.mean(score[ctr_mask]))**2 /
                     (2 * np.std(score[ctr_mask])) )

            self.MSLL = np.mean(SLL)

            self.data['GP_nmodel_pred'] = y_pred
            self.data['GP_nmodel_sigma'] = sigma
            self.data['GP_nmodel_residuals'] = y_true - y_pred

    def svgp_normative_model(self,conf_mat,score,ctr_mask,nu=2.5,batch_size=256,n_inducing=500,num_epochs=20):
        """ Compute SVGP model. See GPyTorch documentation for further details:
        https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html#Creating-a-SVGP-Model.

        Parameters
        -------
        conf_mat: array
            Confounds with categorical values dummy encoded.
        score: array
            Score/response variable.
        ctr_mask: array
            Mask (boolean array) with controls marked True.
        nu: float
            Nu parameter of Matern kernel.
        batch_size: int
            Batch size for SVGP model training and prediction. Default value is 256.
        n_inducing: int
            Number of inducing points for SVGP model. Default value is 500.
        num_epochs: int
            Number of epochs (passes through entire dataset) to train SVGP for. Default value is 20.
        
        Raises
        -------
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
            means, sigmas = svgp.predict()

            y_pred = means.numpy()
            y_true = score
            residuals = (y_true - y_pred).astype(float)

            self.data['GP_nmodel_pred'] = y_pred
            self.data['GP_nmodel_sigma'] = sigmas.numpy()
            self.data['GP_nmodel_residuals'] = residuals
            return svgp.loss

    def _plot(self, plot_type=None):
        """Plot the data with the normative model overlaid.

        Args:
            plot_type (str, optional): type of plot among "LOESS" (local polynomial),
            "Centiles", "GP" (gaussian processes), or "None" (data points only). 
            Defaults to None.

        Returns:
            Axis: handle for the matplotlib axis of the plot.
        """
        ax = sns.scatterplot(data=self.data, x='age', y='score',
                             hue='group', style='group')
        if plot_type == 'LOESS':
            ax.plot(self.bins, self.zm, '-k')
        if plot_type == 'Centiles':
            ax.plot(self.bins, self.z[:, 50], '-k')
        if plot_type == 'GP':
            tmp = self.data.sort_values('age')
            ax.plot(tmp['age'], tmp['GP_nmodel_pred'], '-k')
        return ax

    def plot(self, plot_type=None):
        """Plot the data with the normative model overlaid.

        Args:
            plot_type (str, optional): type of plot among "LOESS"
            (local polynomial), "Centiles", "GP" (gaussian processes),
            or "None" (data points only). Defaults to None.
        """
        plt.figure()
        self._plot(plot_type)
        plt.show()
