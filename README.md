# slearn


[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![PyPI version](https://badge.fury.io/py/slearn.svg)](https://badge.fury.io/py/slearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A package linking symbolic representation with sklearn for time series prediction

Install the slearn package simply by
```
$ pip install slearn
```

## Symbolic machine learning prediction
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

## Prediction with symbolic representation

Load libraries.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from slearn.slearn import *

time_series = pd.read_csv("Amazon.csv") # load the required dataset, here we use Amazon stock daily close price.
ts = time_series.Close.values
```

Set the number of symbols you would like to predict.
```python
step = 200
```

You can select the available classifiers for prediction, and the parameters of the chosen classifier follow the same as the scikit-learn library.
We usually deploy ABBA symbolic representation, since it achieves better forecasting against SAX.

Use Gaussian Naive Bayes method: 
```python
sl = slearn(series=ts, method='ABBA', 
            gap=10, step=step,
            tol=0.01, alpha=0.2, 
            form='numeric', classifier_name="GaussianNB",
            random_seed=1, verbose=1)
sklearn_params = {'var_smoothing':0.001}
abba_nb_pred = sl.predict(**sklearn_params)
```


Use neural network models method: 
```python
sl = slearn(series=ts, method='ABBA',
            gap=10, step=step,
            tol=0.01, alpha=0.2, 
            form='numeric', classifier_name="MLPClassifier",
            random_seed=1, verbose=1)
sklearn_params = {'hidden_layer_sizes':(20,80), 'learning_rate_init':0.1}
abba_nn_pred = sl.predict(**sklearn_params)
```

We can plot the prediction, 

```python
sns.set_theme(style="whitegrid")
plt.figure(figsize=(25, 9))
sns.set(font_scale=2, style="whitegrid")
sns.lineplot(x=np.arange(0, len(ts)), y= ts, color='c', linewidth=6, label='Time series')
sns.lineplot(x=np.arange(len(ts), len(ts)+min_len), y=abba_nb_pred[:min_len], color='tomato', linewidth=6, label='Prediction (ABBA - GaussianNB)')
sns.lineplot(x=np.arange(len(ts), len(ts)+min_len), y=abba_nn_pred[:min_len], color='darkgreen', linewidth=6, label='Prediction (ABBA - MLPClassifier)')
plt.tight_layout()
plt.tick_params(axis='both', labelsize=25)
plt.show()
```


![original image](https://raw.githubusercontent.com/nla-group/slearn/master/doc/demo.png)
