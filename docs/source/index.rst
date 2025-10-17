Welcome to slearn's documentation!
==================================

skearn is a package linking symbolic representation (SAX, ABBA, and fABBA) with scikit-learn machine learning for time series prediction. Symbolic representations of time series have proved their usefulness in the field of time series motif discovery, clustering, classification, forecasting, anomaly detection, etc. Symbolic time series representation methods do not only reduce the dimensionality of time series but also speed up the downstream time series task. It has been demonstrated by [S. Elsworth and S. Güttel, Time series forecasting using LSTM networks: a symbolic approach, arXiv, 2020] that symbolic forecasting has greatly reduced the sensitivity of hyperparameter settings for Long Short Term Memory networks. How to appropriately deploy machine learning algorithms on the level of symbols instead of raw time series poses a challenge to the interest of applications. To boost the development of the research community on symbolic representation, we develop this Python library to simplify the process of machine learning algorithm practice on symbolic representation.

Before getting started, please install the slearn package simply by 


Installation guide
------------------------------
slearn has the following dependencies for its clustering functionality:

   * numpy>=1.21
   * scipy>1.6.0
   * pandas
   * scikit-learn
    
To install the current release via PIP use:

.. parsed-literal::
    
    pip install slearn

.. admonition:: Note
   
   The documentation is still on going.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   symbol_seeds
   symbols_machine_learning
   predict_with_symbols_representation
   dmetric
   api
   license



