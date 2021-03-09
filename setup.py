import os
from setuptools import setup

setup(
    name="pynm",
    version="v0.1-alpha.1",
    author="Annabelle HARVEY, Guillaume DUMAS",
    author_email="annabelle.harvey@umontreal.ca, guillaume.dumas@ppsp.team",
    description=("Normative Modeling in Python"),
    license="BSD",
    keywords="gaussian processes statistics modeling",
    url="https://github.com/ppsp-team/PyNM",
    packages=['pynm', 'test'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: BSD License",
    ],
)
