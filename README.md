# slearn


[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![PyPI version](https://badge.fury.io/py/slearn.svg)](https://badge.fury.io/py/slearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A package linking symbolic representation with sklearn for time series prediction

Install the slearn package simply by
```
$pip install slearn
```

Then, import the package
```python
>>> from slearn import symbolicML, slearn
```

We can predict any symbolic sequence by choosing the classifiers available in scikit-learn.
```python
>>> string = 'aaaabbbccd'
>>> sbml = symbolicML(classifier_name="MLPClassifier", gap=3, random_seed=0, verbose=0)
>>> x, y = sbml._encoding(string)
>>> pred = sbml.forecasting(x, y, step=5, hidden_layer_sizes=(10,10), learning_rate_init=0.1)
>>> print(pred)
['d', 'b', 'a', 'b', 'b'] # the prediction
```

Also, you can use it by passing into parameters of dictionary form
```python
>>> string = 'aaaabbbccd'
>>> sbml = symbolicML(classifier_name="MLPClassifier", gap=3, random_seed=0, verbose=0)
>>> x, y = sbml._encoding(string)
>>> params = {'hidden_layer_sizes':(10,10), 'activation':'relu', 'learning_rate_init':0.1}
>>> pred = sbml.forecasting(x, y, step=5, **params)
>>> print(pred)
['d', 'b', 'a', 'b', 'b'] # the prediction
```
But ensure that parameters are existing in the scikit-learn classifiers.
