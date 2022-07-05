import numpy as np

def read_confounds(confounds):
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

def RMSE(y_true,y_pred):
    """Calculates Root Mean Square Error (RMSE).

    Parameters
    ----------
    y_true: array
        True values for response variable.
    y_pred: array
        Predicted values for response variable
    
    Returns
    -------
    float
        RMSE value for inputs.
    """
    return (np.mean((y_true - y_pred)**2))**0.5

def SMSE(y_true,y_pred):
    """Calculates Standardized Mean Square Error (SMSE).

    Parameters
    ----------
    y_true: array
        True values for response variable.
    y_pred: array
        Predicted values for response variable
    
    Returns
    -------
    float
        SMSE value for inputs.
    """
    return (np.mean((y_true - y_pred)**2))**0.5/np.std(y_true)

def MSLL(y_true,y_pred,sigma,y_train_mean,y_train_sigma):
    """Calculates Mean Standardized Log Loss (MSLL).

    Parameters
    ----------
    y_true: array
        True values for response variable.
    y_pred: array
        Predicted values for response variable
    sigma: array
        Standard deviation of predictive distribution.
    y_train_mean: float
        Mean of training data.
    y_train_sigma: float
        Standard deviation of training data.
    
    
    Returns
    -------
    float
        MSLL value for inputs.
    """
    #SLL = (0.5 * np.log(2 * np.pi * sigma**2) +
    #                (y_true - y_pred)**2 / (2 * sigma**2) -
    #                (y_true - np.mean(y_true))**2 /
    #                (2 * np.std(y_true)))

    # Negative log probability under the model
    model = (0.5*np.log(2*np.pi*sigma**2)) + (y_true - y_pred)**2/(2*sigma**2)

    # Negative log probability under trivial model
    trivial = (0.5*np.log(2*np.pi*y_train_sigma**2)) + (y_true - y_train_mean)**2/(2*y_train_sigma**2)

    SLL = model - trivial
    return np.mean(SLL)