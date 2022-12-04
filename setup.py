import os
from setuptools import setup

setup(
    name="pynm",
    version="1.0.1",
    author="Annabelle HARVEY, Guillaume DUMAS",
    author_email="annabelle.harvey@umontreal.ca, guillaume.dumas@ppsp.team",
    description=("Python implementation of Normative Modelling", 
                 "with GAMLSS, Gaussian Processes, LOESS & Centiles approaches."),
    long_description_content_type="text/x-rst",
    license="BSD",
    keywords="gaussian processes statistics modeling",
    url="https://github.com/ppsp-team/PyNM",
    packages=['pynm', 'test', 'pynm/models'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: BSD License",
    ],
    entry_points={
        'console_scripts': [
            'pynm = pynm.cli:main',
        ],
    },
   install_requires=[
        'gpytorch >= 1.4.0',
        'matplotlib >= 3.3.4',
        'numpy >= 1.19.5',
        'pandas >= 1.1.5',
        'rpy2 >= 3.5.4',
        'scikit_learn >= 1.1.2',
        'scipy >= 1.5.3',
        'seaborn >= 0.12.0',
        'statsmodels >= 0.13.2',
        'torch >= 1.12.1',
        'tqdm >= 4.59.0',
   ],
)
