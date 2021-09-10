import numpy as np
from scipy.stats.mstats import mquantiles

def centiles_fit(train_data,bins,bin_width):
    """ Fit Centiles model.

    Parameters
    ----------
    train_data: array
        Training data for Centiles model.
    bins: array
        Bins for Centiles model.
    bin_width: float
        Width of each bin.

    Returns
    -------
    array
        Centiles for each bin.
    """
    z = np.zeros([bins.shape[0], 101])  # centiles

    for i, bin_center in enumerate(bins):
        mu = np.array(bin_center)  # bin_center value (age or conf)
        bin_mask = (abs(train_data[:, :1] - mu) <
                        bin_width) * 1.  # one hot mask
        idx = [u for (u, v) in np.argwhere(bin_mask)]
        scores = train_data[idx, 1]

        # if more than 2 non NaN values do the model
        if (~np.isnan(scores)).sum() > 2:
            # centiles
            z[i, :] = mquantiles(scores, prob=np.linspace(0, 1, 101), alphap=0.4, betap=0.4)
        else:
            z[i] = np.nan
    
    return z

def centiles_predict(test_data,bins,z):
    """ Predict from Centiles model.

    Parameters
    ----------
    test_data: array
        Test data for Centiles model. Column 0 is confound, 1 is score.
    bins: array
        Bins for Centiles model.
    z: array
        Centiles for each bin.

    Returns
    -------
    array
        Centile within which each subject falls.
    array
        Centiles for each subject.
    """
    dists = [np.abs(conf - bins) for conf in test_data[:,0]]
    idx = [np.argmin(d) for d in dists]
    centiles = np.array([z[i] for i in idx])

    result = np.zeros(centiles.shape[0])
    max_mask = test_data[:,1] >= np.max(centiles, axis=1)
    min_mask = test_data[:,1] < np.min(centiles, axis=1)
    else_mask = ~(max_mask | min_mask)
    result[max_mask] = 100
    result[min_mask] = 0
    result[else_mask] = np.array([np.argmin(test_data[:,1][i] >= centiles[i]) for i in range(test_data.shape[0])])[else_mask]

    return result, centiles