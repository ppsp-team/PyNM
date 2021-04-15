![PyNM Logo](pynm_logo.png)

[![PyPI version shields.io](https://img.shields.io/pypi/v/pynm.svg)](https://pypi.org/project/pynm/) <a href="https://travis-ci.org/ppsp-team/pynm"><img src="https://travis-ci.org/ppsp-team/pynm.svg?branch=master"></a> [![license](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Lightweight Python implementation of Normative Modelling with Gaussian Processes, LOESS & Centiles approaches.

For a more advanced implementation, see the Python librairie [PCNtoolkit](https://github.com/amarquand/PCNtoolkit).

## Roadmap

- [ ] Optimize for large input size with [GPflow](https://github.com/GPflow/GPflow)
- [ ] Addition of the commande line utility (c.f. [post](https://gehrcke.de/2014/02/distributing-a-python-command-line-application/))
- [ ] Coding of key unit tests
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
            [--conf CONF] [--score SCORE] [--group GROUP]

optional arguments:
  -h, --help            show this help message and exit
  --pheno_p PHENO_P     path to phenotype data
  --out_p OUT_P         path to output dir
  --confounds CONFOUNDS
                        list of confounds to use in GP model, formatted as a
                        string with commas between confounds (column names
                        from phenotype dataframe) and categorical confounds
                        marked as C(my_confound).
  --conf CONF           single confound to use in LOESS & centile models
  --score SCORE         response variable, must be column title from phenotype
                        dataframe
  --group GROUP         group, must be column title from phenotype dataframe
                        that indicates probands and controls as 'PROB' and
                        'CTR' or 0 and 1
```
