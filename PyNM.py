#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : PyNM.py
# description     : Compute a Centiles- & LOESS-based normative models
# author          : Guillaume Dumas, Institut Pasteur
# date            : 2019-06-03
# notes           : the input dataframe must contains at least 3 columns:
#                   "group" with controls marked as "CTR" and probands as "PROB"
#                   "age" in days, and "score" which is the measure to model
# licence         : BSD 3-Clause License
# python_version  : 3.6
# ==============================================================================

from tqdm import tqdm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats.mstats import mquantiles

def PyNM(data, min_age=-1, max_age=-1, min_score=-1, max_score=-1, griddef_age = 365 / 8, kernel_age = 365 * 1.5):
   
    if min_age == -1:
        min_age = np.min(data.loc[:, 'age'])
    if max_age == -1:
        max_age = np.max(data.loc[:, 'age'])
    if min_score == -1:
        min_score = np.min(data.loc[:, 'score'])
    if max_score == -1:
        max_score = np.max(data.loc[:, 'score'])

    x = np.arange(min_age,
                    max_age + kernel_age,
                    griddef_age)

    # CTR==True
    sel = data.loc[(data.group == 'CTR'), ['age', 'score']].values
    d = sel[:, :1]

    xcount = np.zeros(x.shape[0])
    zm = np.zeros(x.shape[0])
    zstd = np.zeros(x.shape[0])
    zci = np.zeros([x.shape[0], 2])
    z = np.zeros([x.shape[0], 101])
    for i, xx in enumerate(x):
        mu = np.array(xx)
        w = (abs(d - mu) < kernel_age) * 1.
        idx = [u for (u, v) in np.argwhere(w)]
        YY = sel[idx, 1]
        XX = sel[idx, 0] - mu
        xcount[i] = len(idx)
        if (~np.isnan(YY)).sum()>2:
            mod = sm.WLS(YY, sm.tools.add_constant(XX),
                            missing='drop',
                            weight=w.flatten()[idx],
                            hasconst=True).fit()
            zm[i] = mod.params[0]
            prstd, iv_l, iv_u = wls_prediction_std(mod, [0, 0])
            zstd[i] = prstd
            zci[i, :] = mod.conf_int()[0, :]  # [iv_l, iv_u]
            z[i, :] = mquantiles(YY,
                                    prob=np.linspace(0, 1, 101),
                                    alphap=0.4,
                                    betap=0.4)
        else:
            zm[i] = np.nan
            zci[i] = np.nan
            zstd[i] = np.nan
            z[i] = np.nan

    error_mea, error_med = 0, 0
    rejected = 0
    for i in range(sel.shape[1]):
        idage = np.argmin(np.abs(sel[i, 1] - x))
        error_mea += (sel[i, 0] - zm[idage])**2
        error_med += (sel[i, 0] - z[idage, 50])**2
    error_mea /= sel.shape[1]
    error_med /= sel.shape[1]
    error_mea = error_mea**0.5
    error_med = error_med**0.5


    def bins_num(r):
        """Give the number of ctr used for this age bin."""
        idage = np.argmin(np.abs(r['age'] - x))
        return xcount[idage]


    def loess_normative_model(r):
        """Compute classical normative model."""
        idage = np.argmin(np.abs(r['age'] - x))
        m = zm[idage]
        std = zstd[idage]
        nmodel = (r['score'] - m) / std
        return nmodel


    def centiles_normative_model(r):
        """Compute centiles normative model."""
        idage = np.argmin(np.abs(r['age'] - x))
        centiles = z[idage, :]
        if r['score'] >= max(centiles):
            result = 100
        else:
            if r['score'] < min(centiles):
                result = 0
            else:
                result = np.argmin(r['score'] >= centiles)
        return result


    data.loc[:, 'participants'] = data.apply(bins_num, axis=1)
    data.loc[:, 'LOESS_nmodel'] = data.apply(loess_normative_model,
                                                axis=1)
    data.loc[(data.LOESS_nmodel <= -2), 'LOESS_rank'] = -2
    data.loc[(data.LOESS_nmodel > -2) &
                (data.LOESS_nmodel <= -1), 'LOESS_rank'] = -1
    data.loc[(data.LOESS_nmodel > -1) &
                (data.LOESS_nmodel <= +1), 'LOESS_rank'] = 0
    data.loc[(data.LOESS_nmodel > +1) &
                (data.LOESS_nmodel <= +2), 'LOESS_rank'] = 1
    data.loc[(data.LOESS_nmodel > +2), 'LOESS_rank'] = 2

    data.loc[:, 'Centiles_nmodel'] = data.apply(centiles_normative_model,
                                                axis=1)
    data.loc[(data.Centiles_nmodel <= 5), 'Centiles_rank'] = -2
    data.loc[(data.Centiles_nmodel > 5) &
                (data.Centiles_nmodel <= 25), 'Centiles_rank'] = -1
    data.loc[(data.Centiles_nmodel > 25) &
                (data.Centiles_nmodel <= 75), 'Centiles_rank'] = 0
    data.loc[(data.Centiles_nmodel > 75) &
                (data.Centiles_nmodel <= 95), 'Centiles_rank'] = 1
    data.loc[(data.Centiles_nmodel > 95), 'Centiles_rank'] = 2

    return data, x, xcount, z, zm, zstd, zci
