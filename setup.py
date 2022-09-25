import os
from setuptools import setup

setup(
    name="pynm",
    version="1.0.0b9",
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
        "Development Status :: 4 - Beta",
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
        'matplotlib',
        'numpy',
        'pandas >= 1.1.5',
        'rpy2 >= 3.1.0',
        'scikit_learn >= 0.24.1',
        'scipy >= 1.5.3',
        'seaborn',
        'statsmodels >= 0.13.2',
        'torch >= 1.8.0',
        'tqdm',
   ],
)
