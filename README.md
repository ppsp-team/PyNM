![PyNM Logo](pynm_logo.png)

[![PyPI version shields.io](https://img.shields.io/pypi/v/pynm.svg)](https://pypi.org/project/pynm/) <a href="https://travis-ci.org/ppsp-team/pynm"><img src="https://travis-ci.org/ppsp-team/pynm.svg?branch=master"></a> [![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Lightweight Python implementation of Normative Modelling with Gaussian Processes, LOESS & Centiles approaches.

For a more advanced implementation, see the Python librairie [PCNtoolkit](https://github.com/amarquand/PCNtoolkit).

## Installation

To install pynm:

```bash
$ pip install pynm
```

Alternatively, for development purposes, clone this repository and run:

```bash
$ git clone https://github.com/ppsp-team/PyNM
$ cd PyNM
$ python setup.py develop
```

All code for PyNM is written in Python (Python>=3.5). See [requirements.txt](https://github.com/ppsp-team/PyNM/blob/master/requirements.txt) for a full list of dependencies.

## Command Line Usage
```
usage: pynm [-h] --pheno_p PHENO_P --out_p OUT_P [--confounds CONFOUNDS]
            [--conf CONF] [--score SCORE] [--group GROUP] [--method METHOD]
            [--num_epochs NUM_EPOCHS] [--n_inducing N_INDUCING]
            [--batch_size BATCH_SIZE] [--length_scale LENGTH_SCALE] [--nu NU]

optional arguments:
  -h, --help                        show this help message and exit
  
  --pheno_p PHENO_P                 Path to phenotype data. Data must be in a .csv file.
  
  --out_p OUT_P                     Path to output directory.
  
  --confounds CONFOUNDS             List of confounds to use in the GP model.The list must
                                    formatted as a string with commas between confounds,
                                    each confound must be a column name from the phenotype
                                    .csv file. Categorical confounds must be denoted by
                                    with C(): e.g. 'C(SEX)' for column name 'SEX'. Default
                                    value is 'age'.
                                    
  --conf CONF                       Single numerical confound to use in LOESS & centile
                                    models. Must be a column name from the phenotype .csv
                                    file. Default value is 'age'.
                                    
  --score SCORE                     Response variable for all models. Must be a column
                                    title from phenotype .csv file. Default value is 'score'.
                                    
  --group GROUP                     Column name from the phenotype .csv file that
                                    distinguishes probands from controls. The column must
                                    be encoded with str labels using 'PROB' for probands
                                    and 'CTR' for controls or with int labels using 1 for
                                    probands and 0 for controls. Default value is 'group'.
                                    
  --method METHOD                   Method to use for the GP model. Can be set to
                                    'auto','approx' or 'exact'. In 'auto' mode, the exact
                                    model will be used for datasets smaller than 1000 data
                                    points. SVGP is used for the approximate model. 
                                    See documentation for details. Default value is 'auto'.
                                    
  --num_epochs NUM_EPOCHS           Number of training epochs for SVGP model. 
                                    See documentation for details. Default value is 20.
                                    
  --n_inducing N_INDUCING           Number of inducing points for SVGP model. 
                                    See documentation for details. Default value is 500.
  --batch_size BATCH_SIZE           Batch size for training and predicting from SVGP
                                    model. See documentation for details. Default value is 256.
  --length_scale LENGTH_SCALE       Length scale of Matern kernel for exact model. 
                                    See documentation for details. Default value is 1.
                                    
  --nu NU                           Nu of Matern kernel for exact and SVGP model.
  
  --train_sample TRAIN_SAMPLE       On what subset to train the model, can be 'controls',
                                    'manual', or a value in (0,1]. Default value is 'controls'.
```
## API Example
```python
from pynm.pynm import PyNM

# Initialize pynm w/ data and confounds
m = PyNM(df,'score','group',
        conf = 'age',                           #age confound for LOESS and Centiles model
        confounds = ['age','C(sex)','C(site)']) #multivarite confounds for GP model

# Run models
m.loess_normative_model()
m.centiles_normative_model()
m.gp_normative_model()

# Collect output
data = m.data
```
## Documentation

All the functions have the classical Python DocStrings that you can summon with ```help()```. You can also see the [tutorials](https://github.com/ppsp-team/PyNM/tree/master/tutorials) for documented examples.

### Training sample
By default, the models are fit on all the controls in the dataset and prediction is then done on the entire dataset. The residuals (scores of the normative model) are then calculated as the difference between the actual value and predicted value for each subject. This paradigm is not meant for situations in which the residuals will then be used in a prediction setting, since any train/test split stratified by proband/control will have information from the training set leaked into the test data.

In order to avoid contaminating the test set, in a prediction setting it is important to fit the normative model on a subset of the controls and then leave those out. This is implemented in PyNM with the `--train_sample` flag. It can be used in three ways:
 1. Number in (0,1]
    - This is simplest usage that defines the sample size, PyNM will then select a random sample of the controls and use those as a training group. 
    - The subjects used in the sample are recorded in the column `'train_sample'` of the resulting PyNM.data object. Subjects used in the training sample are encoded as 1s, and the rest as 0s. 
 3. `'manual'`
    - It is also possible to specify exactly which subjects to use as a training group by providing a column in the input data labeled `'train_sample'` encoded the same way.
 5. `'controls'`
    - This is the default setting that will fit the model on all the controls.

### Centiles and LOESS Models
Both the Centiles and LOESS models are non parametric models based local approximations. They accept only a single dependent variable, passed using the `conf` option.

### Gaussian Process Model
Gaussian Process Regression (GPR), which underpins the Gaussian Process Model, can accept an arbitrary number of dependent variables passed using the `confounds` option. 

GPR is very intensive on both memory and time usage. In order to have a scaleable method, we've implemented both an exact model for smaller datasets and an approximate method, recommended for datasets over ~1000 subjects. The method can be specified using the `method` option, it defaults to `auto` in which the approxiamte model will be chosen for datasets over 1000.

#### Exact Model
The exact model implements [scikit-learn](https://scikit-learn.org/stable/index.html)'s Gaussian Process Regressor. The kernel is composed of a constant kernel, a white noise kernel, and a Matern kernel. The Matern kernel has parameters `nu` and `length_scale` that can be specified. The parameter `nu` has special values at 1.5 and 2.5, using other values will significantly increase computation time. See [documentation](https://scikit-learn.org/stable/modules/gaussian_process.html) for an overview of both.

#### Approximate Model
The approximate model implements a Stochastic Variational Gaussian Process (SVGP) model using [GPytorch](https://gpytorch.ai/), with a kernel closely matching the one in the exact model. SVGP is a deep learning technique that needs to be trained on minibatches for a set number of epochs, this can be tuned with the parameters `batch_size` and `num_epoch`. The model speeds up computation by using a subset of the data as inducing points, this can be controlled with the parameter `n_inducing` that defines how many points to use. See [documentation](https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html) for an overview.

## References

Original papers with Gaussian Processes (GP):
- Marquand et al. Biological Psychiatry 2016 ([doi:10.1016/j.biopsych.2015.12.023](https://doi.org/10.1016/j.biopsych.2015.12.023))
- Marquand et al. Molecular Psychiatry 2019 ([doi:10.1038/s41380-019-0441-1](https://doi.org/10.1038/s41380-019-0441-1))

Example of use of the LOESS approach:
- Lefebvre et al. Front. Neurosci. 2018 ([doi:10.3389/fnins.2018.00662](https://doi.org/10.3389/fnins.2018.00662))
- Maruani et al. Front. Psychiatry 2019 ([doi:10.3389/fpsyt.2019.00011](https://doi.org/10.3389/fpsyt.2019.00011))

For the Centiles approach see:
- Bethlehem et al. Communications Biology 2020 ([doi:10.1038/s42003-020-01212-9](https://doi.org/10.1038/s42003-020-01212-9))
- R implementation [here](https://github.com/rb643/Normative_modeling).

For the SVGP model see:
- Hensman et al. [https://arxiv.org/pdf/1411.2005.pdf](https://arxiv.org/pdf/1411.2005.pdf)

## How to report errors

If you spot any bugs :beetle:? Check out the [open issues](https://github.com/ppsp-team/PyNM/issues) to see if we're already working on it. If not, open up a new issue and we will check it out when we can!

## How to contribute

Thank you for considering contributing to our project! Before getting involved, please review our [contribution guidelines](https://github.com/ppsp-team/PyNM/blob/master/CONTRIBUTING.md).

## Support

This work is supported by [Compute Canada](https://computecanada.ca), [IVADO](https://ivado.ca/), and [FRQS](http://www.frqs.gouv.qc.ca/en/).
