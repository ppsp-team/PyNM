#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : PyNM.py
# description     : Compute a Centiles- & LOESS-based normative models
# author          : Guillaume Dumas, Institut Pasteur
# date            : 2019-06-03
# notes           : The input dataframe column passed to --group must either have
#                   controls marked as "CTR" and probands as "PROB", or controls marked as 0 and probands as 1.
#                   The --pheno_p is for the path to the input dataframe.
#                   The --out_p flag is for the path to save the output dataframe, include the filename
#                   formatted as 'filename.csv'. The confounds columns for the gaussian process
#                   model must be specified using the --confounds flag. The confound for the LOESS and centiles
#                   models must be specified using the --conf flag.
# licence         : BSD 3-Clause License
# python_version  : 3.6
# ==============================================================================

from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from argparse import ArgumentParser
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats.mstats import mquantiles

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct

def read_confounds(confounds):
    #Find categorical values in confounds and clean format
    categorical = []
    clean_confounds = []
    for conf in confounds:
        if ((conf[0:2]=='C(') & (conf[-1]==')')):
            categorical.append(conf[2:-1])
            clean_confounds.append(conf[2:-1])
        else:
            clean_confounds.append(conf)
    return clean_confounds,categorical

class PyNM:
    def __init__(self,data,score='score',group='group',conf='age',confounds=['age','C(sex)','C(site)']):
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
        self.error_mea = None
        self.error_med = None
        
        self.set_group_names()
        
    def set_group_names(self):
        """Read whether subjects are labeled CTR/PROB or 0/1 and set accordingly."""
        labels = list(self.data[self.group].unique())
        if ('CTR' in labels) or ('PROB' in labels):
            self.CTR = 'CTR'
            self.PROB = 'PROB'
        else:
            self.CTR = 0
            self.PROB = 1
    
    def get_masks(self):
        ctr = self.data.loc[(self.data[self.group] == self.CTR)]
        ctr_mask = self.data.index.isin(ctr.index)
        probands = self.data.loc[(self.data[self.group] == self.PROB)]
        prob_mask = self.data.index.isin(probands.index)
        return ctr_mask, prob_mask
    
    #Default values for age in days
    def create_bins(self, min_age=-1, max_age=-1, min_score=-1, max_score=-1, bin_spacing = 365 / 8, bin_width = 365 * 1.5):
        if min_age == -1:
            min_age = self.data[self.conf].min()
        if max_age == -1:
            max_age = self.data[self.conf].max()
        if min_score == -1:
            min_score = self.data[self.score].min()
        if max_score == -1:
            max_score = self.data[self.score].max()
        #if max age less than 300 assume age is in years not days
        if max_age < 300:
            bin_spacing = 1/8
            bin_width = 1.5
        
        #define the bins (according to width by age)
        self.bins = np.arange(min_age,max_age + bin_width,bin_spacing)
        
        #format data
        data = self.data[[self.conf, self.score]].to_numpy(dtype=np.float64)
        
        #take the controls
        ctr_mask,_ = self.get_masks()
        ctr = data[ctr_mask]
        
        self.bin_count = np.zeros(self.bins.shape[0])
        self.zm = np.zeros(self.bins.shape[0]) #mean
        self.zstd = np.zeros(self.bins.shape[0]) #standard deviation
        self.zci = np.zeros([self.bins.shape[0], 2]) #confidence interval
        self.z = np.zeros([self.bins.shape[0], 101]) #centiles
        
        for i, bin_center in enumerate(self.bins):
            mu = np.array(bin_center) #bin_center value (age or conf)
            bin_mask = (abs(ctr[:, :1] - mu) < bin_width) * 1. #one hot mask
            idx = [u for (u, v) in np.argwhere(bin_mask)]
            
            scores = ctr[idx,1]
            adj_conf = ctr[idx, 0] - mu #confound relative to bin center
            self.bin_count[i] = len(idx)
            
            #if more than 2 non NaN values do the model
            if (~np.isnan(scores)).sum()>2:
                mod = sm.WLS(scores, sm.tools.add_constant(adj_conf,has_constant='add'),missing='drop',weight=bin_mask.flatten()[idx],hasconst=True).fit()
                self.zm[i] = mod.params[0] #mean
                
                #std and confidence intervals
                prstd, iv_l, iv_u = wls_prediction_std(mod, [0, 0])
                self.zstd[i] = prstd
                self.zci[i, :] = mod.conf_int()[0, :]  # [iv_l, iv_u]
                
                #centiles
                self.z[i, :] = mquantiles(scores,prob=np.linspace(0, 1, 101),alphap=0.4,betap=0.4)
            else:
                self.zm[i] = np.nan
                self.zci[i] = np.nan
                self.zstd[i] = np.nan
                self.z[i] = np.nan
                
        #mean squared error
        self.error_mea, self.error_med = 0, 0
        
        #for age and score (cols of sel)
        for i in range(ctr.shape[1]):
            idage = np.argmin(np.abs(ctr[i, 1] - self.bins))
            self.error_mea += (ctr[i, 0] - self.zm[idage])**2
            self.error_med += (ctr[i, 0] - self.z[idage, 50])**2
        self.error_mea /= ctr.shape[1]
        self.error_med /= ctr.shape[1]
        self.error_mea = self.error_mea**0.5
        self.error_med = self.error_med**0.5
        
        return self.bins, self.bin_count, self.z, self.zm, self.zstd, self.zci
    
    def bins_num(self):
        """Give the number of ctr used for the age bin each participant is in."""
        if (self.error_mea==None):
            self.create_bins()
        dists = [np.abs(conf - self.bins) for conf in self.data[self.conf]]
        idx = [np.argmin(d) for d in dists]
        n_ctr = [self.bin_count[i] for i in idx]
        self.data['participants'] = n_ctr
        return n_ctr
    
    def loess_rank(self):
        self.data.loc[(self.data.LOESS_nmodel <= -2), 'LOESS_rank'] = -2
        self.data.loc[(self.data.LOESS_nmodel > -2) & (self.data.LOESS_nmodel <= -1), 'LOESS_rank'] = -1
        self.data.loc[(self.data.LOESS_nmodel > -1) & (self.data.LOESS_nmodel <= +1), 'LOESS_rank'] = 0
        self.data.loc[(self.data.LOESS_nmodel > +1) & (self.data.LOESS_nmodel <= +2), 'LOESS_rank'] = 1
        self.data.loc[(self.data.LOESS_nmodel > +2), 'LOESS_rank'] = 2

    def loess_normative_model(self):
        """Compute classical normative model."""
        if (self.error_mea==None):
            self.create_bins()
        dists = [np.abs(conf - self.bins) for conf in self.data[self.conf]]
        idx = [np.argmin(d) for d in dists]
        m = np.array([self.zm[i] for i in idx])
        std = np.array([self.zstd[i] for i in idx])
        nmodel = (self.data[self.score] - m) / std
        self.data['LOESS_nmodel'] = nmodel
        self.loess_rank()
        return nmodel
    
    def centiles_rank(self):
        self.data.loc[(self.data.Centiles_nmodel <= 5), 'Centiles_rank'] = -2
        self.data.loc[(self.data.Centiles_nmodel > 5) & (self.data.Centiles_nmodel <= 25), 'Centiles_rank'] = -1
        self.data.loc[(self.data.Centiles_nmodel > 25) & (self.data.Centiles_nmodel <= 75), 'Centiles_rank'] = 0
        self.data.loc[(self.data.Centiles_nmodel > 75) & (self.data.Centiles_nmodel <= 95), 'Centiles_rank'] = 1
        self.data.loc[(self.data.Centiles_nmodel > 95), 'Centiles_rank'] = 2
    
    def centiles_normative_model(self):
        """Compute centiles normative model."""
        if (self.error_mea==None):
            self.create_bins()
        dists = [np.abs(conf - self.bins) for conf in self.data[self.conf]]
        idx = [np.argmin(d) for d in dists]
        centiles = np.array([self.z[i] for i in idx])
        
        result = np.zeros(centiles.shape[0])
        max_mask = self.data[self.score] >= np.max(centiles,axis=1)
        min_mask = self.data[self.score] < np.min(centiles,axis=1)
        else_mask = ~(max_mask | min_mask)
        result[max_mask] = 100
        result[min_mask] = 0        
        result[else_mask] = np.array([np.argmin(self.data[self.score][i] >= centiles[i]) for i in range(self.data.shape[0])])[else_mask]
        self.data['Centiles_nmodel'] = result
        self.centiles_rank()
        return result
        
    def get_conf_mat(self):
        conf_clean,conf_cat = read_confounds(self.confounds)
        conf_mat = pd.get_dummies(self.data[conf_clean],columns=conf_cat,drop_first=True)
        return conf_mat.to_numpy()
    
    def get_score(self):
        return self.data[self.score].to_numpy()

    def gp_normative_model(self,length_scale=1,nu=2.5, method='auto',batch_size=256,n_inducing=500,num_epochs=10):
        """Compute gaussian process normative model.
           length_scale: length scale parameter of Matern kernel
           nu: nu parameter of Matern kernel
           For Matern kernel parameters see scikit-learn documentation https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html."""
        #get proband and control masks
        ctr_mask, prob_mask = self.get_masks()

        #get matrix of confounds
        conf_mat = self.get_conf_mat()

        #Define independent and response variables
        y = self.data[self.score][ctr_mask].to_numpy().reshape(-1,1)
        X = conf_mat[ctr_mask]
        
        score = self.get_score()
        
        if method == 'auto':
            if self.data.shape[0] > 1000:
                approx = False
            else:
                approx = True
        elif method == 'approx':
            approx = True
        elif method == 'exact':
            approx = False
        else:
            raise ValueError('Method must be one of "auto","approx", or "exact".)
        
        if approx == True:
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

            self.data['GP_nmodel_pred'] = y_pred
            self.data['GP_nmodel_sigma'] = sigma
            self.data['GP_nmodel_residuals'] = y_pred - y_true

    def svgp_normative_model(self,conf_mat,score,ctr_mask,nu=2.5,batch_size=256,n_inducing=500,num_epochs=10):
        try:
            from pynm.approx import SVGP
        except:
            print("GPyTorch and it's dependencies must be installed to use the SVGP model.")
        else:
            svgp = SVGP(conf_mat,score,ctr_mask,n_inducing=n_inducing,batch_size=batch_size)
            svgp.train(num_epochs=num_epochs)
            means, sigmas = svgp.predict()

            y_pred = means.numpy()
            y_true = score
            residuals = (y_pred - y_true).astype(float)

            self.data['GP_nmodel_pred'] = y_pred
            self.data['GP_nmodel_sigma'] = sigmas.numpy()
            self.data['GP_nmodel_residuals'] = residuals
            return svgp.loss
