# slearn


[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![PyPI version](https://badge.fury.io/py/slearn.svg)](https://badge.fury.io/py/slearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/slearn/badges/version.svg)](https://anaconda.org/conda-forge/slearn)
[![Documentation Status](https://readthedocs.org/projects/slearn/badge/?version=latest)](https://slearn.readthedocs.io/en/latest/?badge=latest)


### A package linking symbolic representation with scikit-learn machine learning.

----------------------------------------------------------------------------

Symbolic representations of time series have proved their usefulness in the field of time series motif discovery, clustering, classification, forecasting, anomaly detection, etc.  Symbolic time series representation methods do not only reduce the dimensionality of time series but also speedup the downstream time series task. It has been demonstrated by [S. Elsworth and S. Güttel, Time series forecasting using LSTM networks: a symbolic approach, arXiv, 2020] that symbolic forecasting has greatly reduce the sensitivity of hyperparameter settings for Long Short Term Memory networks. How to appropriately deploy machine learning algorithm on the level of symbols instead of raw time series poses a challenge to the interest of applications. To boost the development of research community on symbolic representation, we develop this Python library to simplify the process of machine learning algorithm practice on symbolic representation.  ``slearn`` library provide different API for symbols generation associated with complexity measure and machine learining traning. We will illustrate several topics in detail as below. 

<strong><em> Now let's get started! </em></strong>

Install the slearn package simply by 
#### pip
```
pip install slearn
```

#### conda
```
conda install -c conda-forge slearn
```

To check which version you install, please use:
```
conda list slearn
```


## Model support


slearn currently supports SAX, ABBA, and fABBA symbolic representation, and the machine learning classifiers as below:

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

Our [documentation](https://slearn.readthedocs.io/en/latest/?badge=latest) is available.



## Citation

If you use the function of ``LZWStringLibrary`` in your research, please cite:
```bibtex
@techreport{CCG21,
  title   = {{A comparison of LSTM and GRU networks for learning symbolic sequences}},
  author  = {Cahuantzi, Roberto and Chen, Xinye and G\"{u}ttel, Stefan},
  year    = {2021},
  number  = {arXiv:2107.02248},
  pages   = {12},
  institution = {The University of Manchester},
  address = {UK},
  type    = {arXiv EPrint},
  url     = {https://arxiv.org/abs/2107.02248}
}
```



## License
This project is licensed under the terms of the [MIT license](https://github.com/nla-group/classix/blob/master/LICENSE).


<p align="left">
  <a>
    <img alt="CLASSIX" src="https://raw.githubusercontent.com/nla-group/classix/master/docs/_images/nla_group.png" width="240" />
  </a>
</p>



## Contributing
Contributing to this repo is welcome! We will work through all the pull requests and try to merge into main branch. 


