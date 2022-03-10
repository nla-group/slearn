.. image:: https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master
    :target: https://app.travis-ci.com/github/nla-group/slearn
    :alt: Build Status
.. image:: https://badge.fury.io/py/slearn.svg
    :target: https://badge.fury.io/py/slearn
    :alt: PyPI version
.. image:: https://img.shields.io/pypi/pyversions/slearn.svg
    :target: https://pypi.python.org/pypi/slearn/
    :alt: PyPI pyversions
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://pypi.python.org/pypi/slearn/
    :alt: PyPI pyversions    
.. image:: https://readthedocs.org/projects/slearn/badge/?version=latest
    :target: https://slearn.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

*A package linking symbolic representation with scikit-learn machine learning for time series prediction.*

Symbolic representations of time series have proved their usefulness in the field of time series motif discovery, clustering, classification, forecasting, anomaly detection, etc.  Symbolic time series representation methods do not only reduce the dimensionality of time series but also speedup the downstream time series task. It has been demonstrated by [S. Elsworth and S. Güttel, Time series forecasting using LSTM networks: a symbolic approach, arXiv, 2020] that symbolic forecasting has greatly reduce the sensitivity of hyperparameter settings for Long Short Term Memory networks. How to appropriately deploy machine learning algorithm on the level of symbols instead of raw time series poses a challenge to the interest of applications. To boost the development of research community on symbolic representation, we develop this Python library to simplify the process of machine learning algorithm practice on symbolic representation. 

---------
Install
---------

Install the slearn package simply by

.. code:: bash
    
    pip install slearn

