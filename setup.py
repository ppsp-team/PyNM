import os
from setuptools import setup

setup(
    name="pynm",
    version="1.0.0b1",
    author="Annabelle HARVEY, Guillaume DUMAS",
    author_email="annabelle.harvey@umontreal.ca, guillaume.dumas@ppsp.team",
    description=("Python implementation of Normative Modelling", 
                 "with GAMLSS, Gaussian Processes, LOESS & Centiles approaches."),
    license="BSD",
    keywords="gaussian processes statistics modeling",
    url="https://github.com/ppsp-team/PyNM",
    packages=['pynm', 'test'],
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
        'matplotlib >= 3.3.4',
        'numpy >= 1.18.5',
        'pandas >= 1.1.5',
        'scikit_learn >= 0.24.1',
        'scipy >= 1.5.3',
        'seaborn >= 0.11.1',
        'statsmodels >= 0.12.2',
        'tqdm == 4.59.0'
   ],
)
