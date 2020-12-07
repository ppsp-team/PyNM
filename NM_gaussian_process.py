import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF, DotProduct
from argparse import ArgumentParser
import gc

def cross_validate(X,y,n_splits=5,mus = [0.1,1,10],nus = [0.5,1.5,2.5]):
    kf = KFold(n_splits=n_splits,shuffle=True)
    res = []
    split = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for m in mus:
            for n in nus:
                kernel = ConstantKernel() + WhiteKernel(noise_level=1) + Matern(length_scale=m, nu=n)
                gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
                gp.fit(X_train, y_train)
                y_pred,_ = gp.predict(X_test, return_std=True)
                mse = mean_squared_error(y_test,y_pred)
                res.append([m,n,mse])
                
                #Force trash collection since models can overwhelm memory
                del gp
                gc.collect()
        print('Done split {}/{}.'.format(split,n_splits))
        split = split + 1

    #Average the MSE across param values
    res = pd.DataFrame(res,columns=['length_scale','nu','MSE'])
    mean_df = res.groupby(['length_scale','nu']).mean()
    mean_df.reset_index(inplace=True)
    best_params = mean_df[mean_df['MSE'] == mean_df['MSE'].min()].to_dict('records')

    return best_params[0]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pheno_p",help="path to phenotype data",dest='pheno_p')
    parser.add_argument("--out_p",help="path to save restuls",dest='out_p')
    parser.add_argument("--confounds",help="list of confounds to use in model, formatted as a string with commas between confounds (column names from phenotype dataframe) and categorical confounds marked as C(my_confound).",dest='confounds')
    parser.add_argument("--y",help="response variable, column title from phenotype dataframe",dest='y')
    args = parser.parse_args()
    
    #Find categorical values in confounds and clean format
    confounds = args.confounds.split(',')
    categorical = []
    for conf in confounds:
        if ((conf[0:2]=='C(') & (conf[-1]==')')):
            confounds.remove(conf)
            categorical.append(conf[2:-1])
            confounds.append(conf[2:-1])
        
    print('Loading data...')
    pheno = pd.read_csv(args.pheno_p)
    
    controls = pheno[pheno['group']=='CTR']
    con_mask = pheno.index.isin(controls.index)
    probands = pheno[pheno['group']=='PROB']
    prob_mask = pheno.index.isin(probands.index)
    
    #Define confounds as matrix for prediction, dummy encode categorical variables
    pheno_confounds = pheno[confounds]
    pheno_confounds = pd.get_dummies(pheno_confounds,columns=categorical,drop_first=True)
    pheno_confounds_cols = pheno_confounds.columns.tolist()
    pheno_confounds = pheno_confounds.to_numpy()
    
    #Define independent and response variables
    y = pheno[args.y][con_mask].to_numpy().reshape(-1,1)
    X = pheno_confounds[con_mask]
    
    print('Cross validating to find best kernel parameters...')
    best_params = cross_validate(X,y)
    
    print('Fitting model on controls...')
    kernel = ConstantKernel() + WhiteKernel(noise_level=1) + Matern(length_scale=best_params['length_scale'], nu=best_params['nu'])
    gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, y)

    print('Predicting normative values for probands...')
    X_pred = pheno_confounds[prob_mask]
    y_pred, sigma = gp.predict(X_pred, return_std=True)
    y_true = pheno[args.y][prob_mask].to_numpy().reshape(-1,1)

    print('Saving results...')
    df = pd.DataFrame(np.concatenate([X_pred,y_pred,y_true,sigma.reshape(-1,1)],axis=1),columns=pheno_confounds_cols+['y_pred','y_true','sigma'])
    df.to_csv(args.out_p + '/results_params_l={}_n={}.csv'.format(best_params['length_scale'],best_params['nu']))