# slearn


[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![PyPI version](https://badge.fury.io/py/slearn.svg)](https://badge.fury.io/py/slearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/slearn/badges/version.svg)](https://anaconda.org/conda-forge/slearn)
[![Documentation Status](https://readthedocs.org/projects/slearn/badge/?version=latest)](https://slearn.readthedocs.io/en/latest/?badge=latest)


### A package linking symbolic representation with scikit-learn machine learning.

----------------------------------------------------------------------------

Symbolic representations of time series have proved their usefulness in the field of time series motif discovery, clustering, classification, forecasting, anomaly detection, etc.  Symbolic time series representation methods do not only reduce the dimensionality of time series but also speedup the downstream time series task. It has been demonstrated by [S. Elsworth and S. Güttel, Time series forecasting using LSTM networks: a symbolic approach, arXiv, 2020] that symbolic forecasting has greatly reduce the sensitivity of hyperparameter settings for Long Short Term Memory networks. How to appropriately deploy machine learning algorithm on the level of symbols instead of raw time series poses a challenge to the interest of applications. To boost the development of research community on symbolic representation, we develop this Python library to simplify the process of machine learning algorithm practice on symbolic representation.  ``slearn`` library provide different API for symbols generation associated with complexity measure and machine learining traning. We will illustrate several topics in detail as below. 

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
``skearn`` enables the generation of strings of tunable complexity using the LZW compressing method as base to approximate Kolmogorov complexity. It also contains the tools for the exploration of the hyperparameter space of commonly used RNNs as well as novel ones.

```python
from slearn import *
df_strings = lzw_string_library(symbols=3, complexity=[4, 9], random_state=0)
print(df_strings)
```

Output:
```
  nr_symbols LZW_complexity length       string
0          3              4      4         ACBB
1          3              9     11  CBACBCABABB
```

### Symbolic time seroes representation


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

The following table summarizes the implemented string distance and similarity metrics, each strictly adhering to their formal definitions. All implementations are robust, handle edge cases (e.g., empty strings), and are tested for correctness.

| Metric Name                        | Type       | Key Features                                                                      | Complexity | Primary Use Cases                        |
|------------------------------------|------------|-----------------------------------------------------------------------------------|------------|------------------------------------------|
| Levenshtein Distance               | Distance   | Counts insertions, deletions, substitutions needed to transform strings            | O(n*m)     | Spell-checking, sequence alignment       |
| Hamming Distance                   | Distance   | Counts differing positions in equal-length strings                                | O(n)       | Error detection, fixed-length sequences  |
| Jaro Similarity                    | Similarity | Measures matching characters and transpositions, normalized to [0,1]              | O(n*m)     | Record linkage, fuzzy matching           |
| Jaro-Winkler Distance              | Similarity | Enhances Jaro with prefix weighting, normalized to [0,1]                          | O(n*m)     | Name matching, deduplication             |
| Cosine Similarity                  | Similarity | Compares word frequency vectors, normalized to [0,1]                              | O(n)       | Text similarity, document comparison     |
| Cosine Bigram Similarity           | Similarity | Compares bigram frequency vectors, normalized to [0,1]                            | O(n)       | Character-level text similarity          |
| LCS Distance                       | Distance   | Based on longest common subsequence, distance = len(s1) + len(s2) - 2*len(LCS)    | O(n*m)     | Sequence alignment, diff tools           |
| Dice’s Coefficient                 | Similarity | Measures shared bigram overlap, normalized to [0,1]                               | O(n)       | Short text similarity, fuzzy matching     |
| Smith-Waterman Distance            | Distance   | Local alignment score (match=2, mismatch=-1, gap=-1), distance = -score           | O(n*m)     | Bioinformatics, partial sequence matching |
| Damerau-Levenshtein Distance       | Distance   | Extends Levenshtein with transpositions                                           | O(n*m)     | Typo correction, spell-checking          |

## Notes
- **Complexity**: `n` and `m` are the lengths of the input strings.
- **Type**: Distance metrics return non-negative values (higher = more different); similarity metrics return [0,1] (1 = identical).

```python
from slearn.dmetric import *

print(damerau_levenshtein_distance("cat", "act"))
print(jaro_winkler_distance("martha", "marhta"))
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
