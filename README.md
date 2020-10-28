# PyMallet

## Acknowledgment

This package is a redistributed fork of https://github.com/mimno/PyMallet. Modifications have been made to the code to enable simpler interfacing with [DLATK](https://github.com/dlatk/dlatk); however, the core functionality remains unmodified.

## Installation

This version of PyMallet can be installed with `pip install dlatk-pymallet`.

## Original README

This package provides tools for extracting latent semantic representations of text, particularly probabilistic topic models.

The implementation of LDA uses Gibbs sampling, which is simple but reliable. People often find the resulting models more useful than the stochastic variational algorithm used in Gensim.

To compile:

    python setup.py build_ext --inplace
