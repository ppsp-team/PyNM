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

def centiles_predict(test_data, score, confound, bins,z):
    """ Predict from Centiles model.

    Parameters
    ----------
    bins: array
    confound: str
    score: str
    test_data: DataFrame
    z: array

    Returns
    -------
    array
        result
    array
        centiles
    array
        centiles_32
    array
        centiles_50
    array
        centiles_68
    """
    dists = [np.abs(conf - bins) for conf in test_data[confound]] #TODO: convert test_data to array for consistency w/ fit
    idx = [np.argmin(d) for d in dists]
    centiles = np.array([z[i] for i in idx])
    centiles_50 = np.array([centiles[i, 50] for i in range(test_data.shape[0])])
    centiles_68 = np.array([centiles[i, 68] for i in range(test_data.shape[0])])
    centiles_32 = np.array([centiles[i, 32] for i in range(test_data.shape[0])])

    result = np.zeros(centiles.shape[0])
    max_mask = test_data[score] >= np.max(centiles, axis=1)
    min_mask = test_data[score] < np.min(centiles, axis=1)
    else_mask = ~(max_mask | min_mask)
    result[max_mask] = 100
    result[min_mask] = 0
    result[else_mask] = np.array([np.argmin(test_data[score][i] >= centiles[i]) for i in range(test_data.shape[0])])[else_mask]

    return result, centiles, centiles_32, centiles_50, centiles_68