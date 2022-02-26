# slearn


[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![PyPI version](https://badge.fury.io/py/slearn.svg)](https://badge.fury.io/py/slearn)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/slearn.svg)](https://pypi.python.org/pypi/slearn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A package linking symbolic representation with scikit-learn machine learning for time series prediction.

Symbolic representations of time series have proved their usefulness in the field of time series motif discovery, clustering, classification, forecasting, anomaly detection, etc.  Symbolic time series representation methods do not only reduce the dimensionality of time series but also speedup the downstream time series task. It has been demonstrated by [S. Elsworth and S. Güttel, Time series forecasting using LSTM networks: a symbolic approach, arXiv, 2020] that symbolic forecasting has greatly reduce the sensitivity of hyperparameter settings for Long Short Term Memory networks. How to appropriately deploy machine learning algorithm on the level of symbols instead of raw time series poses a challenge to the interest of applications. To boost the development of research community on symbolic representation, we develop this Python library to simplify the process of machine learning algorithm practice on symbolic representation.

<strong><em> Now let's get started! </em></strong>

Install the slearn package simply by
```
$ pip install slearn
```
|  Support Classifiers | Parameter call |
|  ----  | ----  |
| [Multi-layer Perceptron](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron)   |'MLPClassifier' |
| [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)  | 'KNeighborsClassifier' |
| [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)   | 'GaussianNB'|
| [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)  | 'DecisionTreeClassifier' |
| [Support Vector Classification](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) | 'SVC' |
| [Radial-basis Function Kernel](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html) | 'RBF'|
| [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  | 'LogisticRegression' |
| [Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)  | 'QuadraticDiscriminantAnalysis' |
| [AdaBoost classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)  | 'AdaBoostClassifier' |
| [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)  | 'RandomForestClassifier' |
| [LightGBM](https://lightgbm.readthedocs.io/en/latest/) | 'LGBM' |

## Symbolic machine learning prediction
Import the package
```python
from slearn import symbolicML
```

We can predict any symbolic sequence by choosing the classifiers available in scikit-learn.
```python
string = 'aaaabbbccd'
sbml = symbolicML(classifier_name="MLPClassifier", ws=3, random_seed=0, verbose=0)
x, y = sbml.encode(string)
pred = sbml.forecast(x, y, step=5, hidden_layer_sizes=(10,10), learning_rate_init=0.1)
print(pred) #  ['d', 'b', 'a', 'b', 'b'] 
```

Also, you can use it by passing into parameters of dictionary form
```python
string = 'aaaabbbccd'
sbml = symbolicML(classifier_name="MLPClassifier", ws=3, random_seed=0, verbose=0)
x, y = sbml.encode(string)
params = {'hidden_layer_sizes':(10,10), 'activation':'relu', 'learning_rate_init':0.1}
pred = sbml.forecast(x, y, step=5, **params)
print(pred) # ['d', 'b', 'a', 'b', 'b'] # the prediction
```
The parameter settings for the chosen classifier follow the same as the scikit-learn library, so just ensure that parameters are existing in the scikit-learn classifiers. More details are refer to scikit-learn website.

## Time series forecasting with symbolic representation

Load libraries.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from slearn import *

time_series = pd.read_csv("Amazon.csv") # load the required dataset, here we use Amazon stock daily close price.
ts = time_series.Close.values
```

Set the number of symbols you would like to predict.
```python
step = 50
```

You can select the available classifiers and symbolic representation method (currently we support SAX and ABBA) for prediction. Similarly, the parameters of the chosen classifier follow the same as the scikit-learn library. We usually deploy ABBA symbolic representation, since it achieves better forecasting against SAX.

slean leverages user-friendly API, time series forecasting follows:

- Step 1: Define the windows size (features size), the forecasting steps, symbolic representation method (SAX or fABBA) and classifier.

- Step 2: Transform time series into symbols with user specified parameters defined for symbolic representation.

- Step 3: Define the classifier parameters and forecast the future values.

Use Gaussian Naive Bayes method: 

```python
sl = slearn(method='fABBA',  ws=3, step=step, classifier_name="GaussianNB") # step 1
sl.set_symbols(series=ts, tol=0.01, alpha=0.2) # step 2
sklearn_params = {'var_smoothing':0.001} # step 3
abba_nb_pred = sl.predict(**sklearn_params) # step 3
```

For the last two lines, they can also be replaced with the alternative way in a clear form:
```python
abba_nb_pred = sl.predict(var_smoothing=0.001) # step 3
```
This follows the same as below.

Try neural network models method: 
```python
sl = slearn(method='fABBA', ws=3, step=step, classifier_name="MLPClassifier") # step 1
sl.set_symbols(series=ts, tol=0.01, alpha=0.2)  # step 2
sklearn_params = {'hidden_layer_sizes':(20,80), 'learning_rate_init':0.1} # step 3
abba_nn_pred = sl.predict(**sklearn_params) # step 3
```



Now we try to preduct real-world time series. We can plot the prediction and compare the results. 

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from slearn import * 
np.random.seed(0)

time_series = pd.read_csv("Amazon.csv")
ts = time_series.Close.values
length = len(ts)
train, test = ts[:round(0.9*length)], ts[round(0.9*length):]

sl = slearn(method='fABBA', ws=8, step=1000, classifier_name="GaussianNB")
sl.set_symbols(series=train, tol=0.01, alpha=0.1) 
abba_nb_pred = sl.predict(var_smoothing=0.001)
sl = slearn(method='fABBA', ws=8, step=1000, classifier_name="DecisionTreeClassifier")
sl.set_symbols(series=train, tol=0.01, alpha=0.1) 
abba_nn_pred = sl.predict(max_depth=10, random_state=0)
sl = slearn(method='fABBA', ws=8, step=1000, classifier_name="KNeighborsClassifier")
sl.set_symbols(series=train, tol=0.01, alpha=0.1) 
abba_kn_pred = sl.predict(n_neighbors=10)
sl = slearn(method='fABBA', ws=8, step=100, classifier_name="SVC")
sl.set_symbols(series=train, tol=0.01, alpha=0.1) 
abba_svc_pred = sl.predict(C=20)
min_len = np.min([len(test), len(abba_nb_pred), len(abba_nn_pred)])

plt.figure(figsize=(20, 5))
sns.set(font_scale=1.5, style="whitegrid")
sns.lineplot(data=test[:min_len], linewidth=6, color='k', label='ground truth')
sns.lineplot(data=abba_nb_pred[:min_len], linewidth=6, color='tomato', label='prediction (ABBA - GaussianNB)')
sns.lineplot(data=abba_nn_pred[:min_len], linewidth=6, color='m', label='prediction (ABBA - DecisionTreeClassifier)')
sns.lineplot(data=abba_nn_pred[:min_len], linewidth=6, color='c', label='prediction (ABBA - KNeighborsClassifier)')
sns.lineplot(data=abba_svc_pred[:min_len], linewidth=6, color='yellowgreen', label='prediction (ABBA - Support Vector Classification)')
plt.legend()
plt.tick_params(axis='both', labelsize=15)
plt.show()
```
![original image](https://raw.githubusercontent.com/nla-group/slearn/master/docs/demo1.png)


## Flexible symbolic sequence generator
slearn library also contains functions for the generation of strings of tunable complexity using the LZW compressing method as base to approximate Kolmogorov complexity.


```python
from slearn import *
df_strings = LZWStringLibrary(symbols=3, complexity=[3, 9])
df_strings
```
Processing: 2 of 2
 ||nr_symbols | LZW_complexity | length | string |
|:---:| ----------:| --------------:| -------:| ------:|
| 0 | 3 | 3 | 3 | BCA |
| 1 | 3 | 9 | 12 | ABCBBCBBABCC |
```python
df_iters = pd.DataFrame()
for i, string in enumerate(df_strings['string']):
    kwargs = df_strings.iloc[i,:-1].to_dict()
    seed_string = df_strings.iloc[i,-1]
    df_iter = RNN_Iteration(seed_string, iterations=2, architecture='LSTM', **kwargs)
    df_iter.loc[:, kwargs.keys()] = kwargs.values()
    df_iters = df_iters.append(df_iter)
df_iter.reset_index(drop=True, inplace=True)
```
...
```python
df_iters.reset_index(drop=True, inplace=True)
df_iters
```
 || jw | dl | total_epochs | seq_test | seq_forecast | total_time | nr_symbols | LZW_complexity | length |
|:---:| --------:| --------:| --------:| --------------:| --------------:| --------:| ---:| ---:| ---:|
|0 |1.000000	|1.0	|12	|ABCABCABCA	|ABCABCABCA	|2.685486	|3	|3	|3|
|1	|1.000000	|1.0	|14	|ABCABCABCA	|ABCABCABCA	|2.436733	|3	|3	|3|
|2	|0.657143	|0.5	|36	|CBBCBBABCC	|AABCABCABC	|3.352712	|3	|9	|12|
|3	|0.704762	|0.4	|36	|CBBCBBABCC	|ABCBABBBBB	|3.811584	|3	|9	|12|

