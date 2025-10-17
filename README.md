# slearn: Python package for learning symbolic sequences


[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![Workflow for Codecov](https://github.com/nla-group/slearn/actions/workflows/unittests.yml/badge.svg)](https://github.com/nla-group/slearn/actions/workflows/unittests.yml)
[![PyPI version](https://badge.fury.io/py/slearn.svg)](https://badge.fury.io/py/slearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/slearn/badges/version.svg)](https://anaconda.org/conda-forge/slearn)
[![Documentation Status](https://readthedocs.org/projects/slearn/badge/?version=latest)](https://slearn.readthedocs.io/en/latest/?badge=latest)


Symbolic representations of time series have demonstrated their effectiveness in tasks such as motif discovery, clustering, classification, forecasting, and anomaly detection. These methods not only reduce the dimensionality of time series data but also accelerate downstream tasks.
Elsworth and Güttel [Time Series Forecasting Using LSTM Networks: A Symbolic Approach, arXiv, 2020] have shown that symbolic forecasting significantly reduces the sensitivity of Long Short-Term Memory (LSTM) networks to hyperparameter settings. However, deploying machine learning algorithms at the symbolic level—rather than on raw time series data—remains a challenging problem for many practical applications. 


To support the research community and streamline the application of machine learning to symbolic representations, we developed the ``slearn`` Python library. This library offers APIs for symbolic sequence generation, complexity measurement, and machine learning model training on symbolic data. We will illustrate several core features and use cases below.

## Install

<strong><em> Install the slearn package simply by  </em></strong>

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

## Usage

### Generate strings with customized complexity

A key feature of ``skearn`` is its ability to compute distances between symbolic sequences, enabling similarity or dissimilarity measurements after transformation. The library includes the ``LZWStringLibrary``, which supports string distance computation based on Lempel-Ziv-Welch (LZW) complexity. 

``skearn`` enables the generation of strings of tunable complexity using the LZW compressing method as base to approximate Kolmogorov complexity. It also contains the tools for the exploration of the hyperparameter space of commonly used RNNs as well as novel ones.
The ``skearn`` library uses the LZWStringLibrary to compute distances between symbolic sequences. The distance measure is based on the LZW complexity, which quantifies the complexity of a string by counting the number of unique substrings in its LZW compression dictionary. The library provides a method called distance in the LZWStringLibrary class to compute the distance between two strings, which can be used to compare symbolic representations of time series.
The distance measure is typically normalized and leverages the LZW complexity to provide a similarity score between two sequences. This is particularly useful when comparing time series that have been transformed into symbolic sequences using methods like SAX.


```python
from slearn import *

str_, str_complex = lzw_string_generator(2, 3, priorise_complexity=True, random_state=2)
print(f"string: {str_}, complexity: {str_complex}")

str_, str_complex = lzw_string_generator(2, 3, priorise_complexity=False, random_state=2)
print(f"string: {str_}, complexity: {str_complex}")

df_strings = lzw_string_seeds(symbols=[2, 3], complexity=[3, 6, 7], priorise_complexity=False, random_state=0)
print(df_strings)
```

Output:
```
string: BAA, complexity: 3
string: BAB, complexity: 3
  nr_symbols LZW_complexity length       string
0          2              3      3          ABA
1          2              6      8     BABBABBA
2          2              7     11  BAAABABAAAA
3          3              3      3          BAC
4          3              6      6       ABCACB
5          3              7      8     ABCAAABB
```


### Benchmarking Transformers and RNNs performance for memorizing capability
``slean`` offers benchmarking tools for compare deep models ability to memorize. It will automatically generated analyzed documents and visualization for tested models via interface ``benchmark_models``. One can either use built-in models or design their own models following [model examples](https://github.com/nla-group/slearn/tree/master/slearn/deep_models.py).  The example can be viewed below.  
```python
from slearn.deep_models import (LSTMModel, GRUModel, TransformerModel, GPTLikeModel)
 # use built-in models or customized your own models. 
from slearn.simulation import benchmark_models

model_list = [LSTMModel, GRUModel, TransformerModel, GPTLikeModel] 
benchmark_models(model_list, 
                  symbols_list=[2, 4, 6, 8],  # number of distinctive numbers
                  complexities=[210, 230, 250, 270, 290], # complexity, the higher complexity indicates a tougher task
                  sequence_lengths=[3500],
                  window_size=100,
                  validation_length=100,
                  stopping_loss=0.1,
                  max_epochs=999,
                  num_runs=5,
                  units=[128],
                  layers=[1, 2, 3],
                  batch_size=256,
                  max_strings_per_complexity=1000,
                  learning_rates=[1e-3, 1e-4]
            ) 
```

### Symbolic time seroes representation


The following table summarizes the implemented Symbolic Aggregate Approximation (SAX) variants and the ABBA method for time series representation:

| Algorithm | Time Series Type | Segmentation | Features Extracted | Symbolization | Reconstruction |
|-----------|------------------|--------------|--------------------|---------------|----------------|
| **SAX**   | Univariate       | Fixed-size segments | Mean (PAA) | Gaussian breakpoints, single symbol per segment | Piecewise constant from PAA values |
| **SAX-TD** | Univariate       | Fixed-size segments | Mean (PAA), slope | Mean to symbol, trend suffix ('u', 'd', 'f') | Linear trends from PAA and slopes |
| **eSAX**  | Univariate       | Fixed-size segments | Min, mean, max | Three symbols per segment (min, mean, max) | Quadratic interpolation from min, mean, max |
| **mSAX**  | Multivariate     | Fixed-size segments | Mean per dimension | One symbol per dimension per segment | Piecewise constant per dimension |
| **aSAX**  | Univariate       | Adaptive segments (based on local variance) | Mean (PAA) | Gaussian breakpoints, single symbol per segment | Piecewise constant from adaptive segments |
| **ABBA**  | Univariate       | Adaptive piecewise linear segments | Length, increment | Clustering (k-means), symbols assigned to clusters | Piecewise linear from cluster centers |

- **SAX**: Standard SAX with fixed-size segments and mean-based symbolization.
- **SAX-TD**: Extends SAX with trend information (up, down, flat) per segment.
- **eSAX**: Enhanced SAX capturing min, mean, and max per segment for smoother reconstruction.
- **mSAX**: Multivariate SAX, processing each dimension independently.
- **aSAX**: Adaptive SAX, adjusting segment sizes based on local variance for better representation of variable patterns.
- **ABBA**: Adaptive Brownian Bridge-based Aggregation, using piecewise linear segmentation and k-means clustering for symbolization (based on https://github.com/nla-group/fABBA).

```python
from slearn.symbols import *


def test_sax_variant(model, ts, t, name, is_multivariate=False):
    symbols = model.fit_transform(ts)
    recon = model.inverse_transform()
    print(f"{name} reconstructed length: {len(recon)}")
    rmse = np.sqrt(np.mean((ts - recon) ** 2))
    return rmse

# Generate test time series
np.random.seed(42)
t = np.linspace(0, 10, 100)
ts = np.sin(t) + np.random.normal(0, 0.1, 100)  # Univariate, main test
ts_multi = np.vstack([np.sin(t), np.cos(t)]).T + np.random.normal(0, 0.1, (100, 2))  # Multivariate


sax = SAX(window_size=10, alphabet_size=8)
rmse = test_sax_variant(sax, ts, t, "SAX")


saxtd = SAXTD(window_size=10, alphabet_size=8)
rmse = test_sax_variant(saxtd, ts, t, "SAX-TD")

    
esax = ESAX(window_size=10, alphabet_size=8)
rmse = test_sax_variant(esax, ts, t, "eSAX")

    
msax = MSAX(window_size=10, alphabet_size=8)
rmse = test_sax_variant(msax, ts_multi, t, "mSAX", is_multivariate=True)

    
asax = ASAX(n_segments=10, alphabet_size=8)
rmse = test_sax_variant(asax, ts, t, "aSAX")
```

### String distance and similarity metrics

``slearn`` includes the implemented interface for string distance and similarity metrics as well as their normalized implementations, each strictly adhering to their formal definitions.
```python
from slearn.dmetric import *

print(damerau_levenshtein_distance("cat", "act"))
print(jaro_winkler_distance("martha", "marhta"))

print(normalized_damerau_levenshtein_distance("cat", "act"))
print(normalized_jaro_winkler_distance("martha", "marhta"))

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

Our [documentation](https://slearn.readthedocs.io/en/latest/?badge=latest) is available.

## Citation
This slearn implementation is maintained by Roberto Cahuantzi (University of Manchester), Xinye Chen (Charles University Prague),  and Stefan Güttel (University of Manchester). If you use the function of ``LZWStringLibrary`` in your research, or if you find slearn useful in your work, please consider citing the paper below. If you have any problems or questions, just drop us an email.

 
```bibtex
@InProceedings{10.1007/978-3-031-37963-5_53,
    author="Cahuantzi, Roberto
    and Chen, Xinye
    and G{\"u}ttel, Stefan",
    title="A Comparison of LSTM and GRU Networks for Learning Symbolic Sequences",
    booktitle="Intelligent Computing",
    year="2023",
    publisher="Springer Nature Switzerland",
    pages="771--785"
}
```



## License
This project is licensed under the terms of the [MIT license](https://github.com/nla-group/classix/blob/master/LICENSE).



## Contributing
Contributing to this repo is welcome! We will work through all the pull requests and try to merge into main branch. 

TO DO LIST:
* language modeling functionalities
* comphrehensive documentation
* performance optimization 
