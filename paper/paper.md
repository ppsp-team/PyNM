---
title: 'PyNM: a Lightweight Python implementation of Normative Modeling.'
tags:
  - Python
  - Normative Modeling
  - Heterogeneity
  - Heteroskedasticity
  - Big Data
  - Gaussian Process
  - GAMLSS
  - Computational Psychiatry
  - Neuroscience
authors:
  - name: Harvey, Annabelle
    orcid: 0000-0002-9940-8799
    affiliation: "1, 2"
  - name: Dumas, Guillaume
    orcid: 0000-0002-2253-1844
    affiliation: "1, 3"
affiliations:
 - name: Centre de Recherche du CHU Sainte-Justine, Université de Montréal, QC, Canada
   index: 1
 - name: Centre de Recherche de l’Institut Universitaire de Gériatrie de Montréal, Université de Montréal, QC, Canada
   index: 2
 - name: Mila - Quebec AI Institute, Université de Montréal, QC, Canada
   index: 3
date: 10 March 2022
bibliography: paper.bib
---


# Summary

The majority of studies in neuroimaging and psychiatry are focussed on case-control analysis [@marquand:2019]. However, case-control relies on well defined groups which is more the exception than the rule in biology. Psychiatric conditions are diagnosed based on symptoms alone, which makes for heterogeneity at the biological level [@marquand:2016]. Relying on mean differences obscures this heterogeneity and the resulting loss of information can produce unreliable results or misleading conclusions [@loth:2021].

Normative Modeling is an emerging alternative to case-control analyses that seeks to parse heterogeneity by looking at how individuals deviate from the normal trajectory. Analogous to normative growth charts, normative models map the mean and variance of a trait for a given population against a set of explanatory variables (usually including age). Statistical inferences at the level of the individual participant can then be obtained with respect to the normative range [@marquand:2019]. This framework can detect patterns of abnormality that might not be consistent across the population and recasts disease as an extreme of the normal range.

PyNM is a lightweight python implementation of Normative Modeling making it approachable and easy to adopt. The package provides:

- Python API and a command-line interface for wide accessibility
- Automatic dataset splitting and cross-validation
- Five models from various back-ends in a unified interface that cover a broad range of common use cases
- Solutions for very large datasets and heteroskedastic data
- Integrated plotting and evaluation functions to quickly check the validity of the model fit and results
- Comprehensive and interactive tutorials


# Statement of need

The basic idea underpinning Normative Modeling is to fit a model on the controls (or a subset of them) of a dataset, and then apply it to the rest of the participants. The difference between the model’s prediction and the ground truth for the unseen participants relative to the variance around the prediction quantifies their deviation from the normal. While simple in concept, implementing Normative Modeling requires some care in managing the dataset and choosing an appropriate model.

In principle, any model that estimates both the mean and variance of the predictive distribution could be used for Normative Modeling. However, in practice we impose more constraints. First and foremost, the assumptions of the model must be met by the data. Second, we want to distinguish between epistemic and aleatoric uncertainty. Epistemic or systematic uncertainty stems from how information about the distribution is collected, whereas aleatoric uncertainty is intrinsic to the distribution and represents the true variation of the population [@xu:2021].

To the author’s knowledge, PCNtoolkit [@pcntoolkit] is the only other available package for Normative Modeling. It implements methods that have been applied in a range of psychiatry and neuroimaging studies including [@kia:2020] and [@fraza:2021], and is accompanied by thorough tutorials and a framework for Normative Modeling in computational psychiatry [@rutherford:2021]. While it includes features that make it an obvious choice for advanced users in many cases, is not as approachable to beginners and does not implement several key models.

PyNM is intended to take users through their first steps in Normative Modeling to using advanced models on complex datasets. Crucially, it manages the dataset and has interactive tutorials – making it quick for new users to try the method either on their own data or on provided simulated data. The tutorials motivate the use of each model and highlight their limitations to help clarify which model is appropriate for what data, and built in plotting and evaluation functions \autoref{fig:Figure 1} make it simple to check the validity of the model output. The package includes five models from various backends in a unified interface, including a wrapper for GAMLSS [@rigby:2005] from R that is otherwise not yet available in python, and the selected models cover many settings including big data and heteroskedasticity.

Earlier versions of PyNM code were used in the following publications:

- @lefebvre:2018
- @maruani:2019
- @bethlehem:2020

# Usage Example
```
from pynm.pynm import PyNM

# Load data
# df contains columns ‘score’,’group’,’age’,’sex’,’site’
df = pd.read_csv(‘data.csv’)

# Initialize pynm w/ data and confounds
m = PyNM(df,'score','group', confounds = ['age','c(sex)','c(site)'])

# Run models
m.loess_normative_model()
m.centiles_normative_model()
m.gp_normative_model()
m.gamlss_normative_model()

# Collect output
data = m.data
```

# Figures

![Output of built in plotting function for model fit and residuals.\label{fig:Figure 1}](figure1.png)

# Acknowledgements

The development of this code has benefited from useful discussions with Andre Marquand, Thomas Wolfers, Eva Loth, Jumana Amad, Richard Bethelem, and Michael Lombardo.

Fundings: GD is supported by IVADO, FRQS, CFI, MITACS, and Compute Canada.

# References