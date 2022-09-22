import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def loess_fit(train_data,bins,bin_width):
    """ Fit LOESS model.

    Parameters
    ----------
    train_data: array
        Training data for LOESS model. Column 0 is confound, 1 is score.
    bins: array
        Bins for LOESS model.
    bin_width: float
        Width of each bin.

    Returns
    -------
    array
        Mean of each bin.
    array
        Standard deviation of each bin.
    array
        Confidence interval of each bin.
    """
    zm = np.zeros(bins.shape[0])  # mean
    zstd = np.zeros(bins.shape[0])  # standard deviation
    zci = np.zeros([bins.shape[0], 2])  # confidence interval

    for i, bin_center in enumerate(bins):
        mu = np.array(bin_center)  # bin_center value (age or conf)
        bin_mask = (abs(train_data[:, :1] - mu) < bin_width) * 1.
        idx = [u for (u, v) in np.argwhere(bin_mask)]

        scores = train_data[idx, 1]
        adj_conf = train_data[idx, 0] - mu  # confound relative to bin center

        # if more than 2 non NaN values do the model
        if (~np.isnan(scores)).sum() > 2:
            mod = sm.WLS(scores, sm.tools.add_constant(adj_conf, 
                                                        has_constant='add'),
                            missing='drop', weights=bin_mask.flatten()[idx],
                            hasconst=True).fit()
            zm[i] = mod.params[0]  # mean

            # std and confidence intervals
            prstd,_,_ = wls_prediction_std(mod, [0, 0])
            zstd[i] = prstd
            zci[i, :] = mod.conf_int()[0, :]  # [iv_l, iv_u]

        else:
            zm[i] = np.nan
            zci[i] = np.nan
            zstd[i] = np.nan
    
    return zm, zstd, zci

def loess_predict(test_data,bins,zm,zstd):
    """ Predict from LOESS model.

    Parameters
    ----------
    test_data: array
        Test data for LOESS model. Column 0 is confound, 1 is score.
    bins: array
        Bins for LOESS model.
    zm: array
        Mean of each bin.
    zstd: array
        Standard deviation of each bin.

    Returns
    -------
    array
        Mean for each subject.
    array
        Standard deviation for each subject.
    """
    dists = [np.abs(conf - bins) for conf in test_data[:,0]]
    idx = [np.argmin(d) for d in dists]
    m = np.array([zm[i] for i in idx])
    std = np.array([zstd[i] for i in idx])

    return m, std