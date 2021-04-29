![PyNM Logo](pynm_logo.png)

[![PyPI version shields.io](https://img.shields.io/pypi/v/pynm.svg)](https://pypi.org/project/pynm/) <a href="https://travis-ci.org/ppsp-team/pynm"><img src="https://travis-ci.org/ppsp-team/pynm.svg?branch=master"></a> [![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Lightweight Python implementation of Normative Modelling with Gaussian Processes, LOESS & Centiles approaches.

For a more advanced implementation, see the Python librairie [PCNtoolkit](https://github.com/amarquand/PCNtoolkit).

## Roadmap

- [x] Optimize for large input size with [GPyTorch](https://gpytorch.ai/)
- [x] Addition of the commande line utility (c.f. [post](https://gehrcke.de/2014/02/distributing-a-python-command-line-application/))
- [x] Coding of key unit tests
- [ ] Creation of a clear tutorial
- [ ] Documentation of all the functions
- [ ] Submission to JO

## References

Original papers with Gaussian Processes (GP):
- [Marquand et al. Biological Psychiatry 2016](https://www.sciencedirect.com/science/article/pii/S0006322316000020)
- [Marquand et al. Molecular Psychiatry 2019](https://www.nature.com/articles/s41380-019-0441-1)

Example of use of the LOESS approach:
- [Lefebvre et al. Front. Neurosci. 2018](https://www.frontiersin.org/articles/10.3389/fnins.2018.00662/full)
- [Maruani et al. Front. Psychiatry 2019](https://www.frontiersin.org/articles/10.3389/fpsyt.2019.00011/full)

See also [Bethlehem et al. Communications Biology 2020](https://www.nature.com/articles/s42003-020-01212-9) with R implementation [here](https://github.com/rb643/Normative_modeling).

## Usage
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
