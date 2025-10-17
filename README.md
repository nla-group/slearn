# slearn: Python Package for Learning Symbolic Sequences

[![Build Status](https://app.travis-ci.com/nla-group/slearn.svg?token=SziD2n1qxpnRwysssUVq&branch=master)](https://app.travis-ci.com/github/nla-group/slearn)
[![Codecov Workflow](https://github.com/nla-group/slearn/actions/workflows/unittests.yml/badge.svg)](https://github.com/nla-group/slearn/actions/workflows/unittests.yml)
[![PyPI Version](https://badge.fury.io/py/slearn.svg)](https://badge.fury.io/py/slearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conda Version](https://anaconda.org/conda-forge/slearn/badges/version.svg)](https://anaconda.org/conda-forge/slearn)
[![Documentation Status](https://readthedocs.org/projects/slearn/badge/?version=latest)](https://slearn.readthedocs.io/en/latest/?badge=latest)

## Overview

The `slearn` Python package is designed for learning and processing symbolic sequences, particularly for time series analysis. Symbolic representations reduce the dimensionality of time series data, accelerating tasks such as motif discovery, clustering, classification, forecasting, and anomaly detection. As demonstrated by Elsworth and Güttel ([arXiv, 2020](https://arxiv.org/abs/2003.11280)), symbolic forecasting reduces the sensitivity of Long Short-Term Memory (LSTM) networks to hyperparameter settings, making it a powerful approach for machine learning on symbolic data.

`slearn` provides APIs for:
- Generating symbolic sequences with controlled complexity using Lempel-Ziv-Welch (LZW) compression.
- Computing distances between symbolic sequences for similarity analysis.
- Benchmarking deep learning models (e.g., LSTMs, GRUs, Transformers) for sequence memorization.
- Supporting symbolic time series representations like SAX and ABBA variants.

This package is ideal for researchers and practitioners working on symbolic time series analysis and machine learning.

## Installation

Install `slearn` using either pip or conda:

### pip
```bash
pip install slearn
```

### conda
```bash
conda install -c conda-forge slearn
```

To verify the installed version:
```bash
pip show slearn
# or
conda list slearn
```

**Dependencies**:
- Python 3.6+
- NumPy
- pandas
- scikit-learn
- tqdm (optional, for progress tracking)

## Key Features

### 1. Generating Strings with Controlled Complexity

The `LZWStringLibrary` module generates strings with specified numbers of unique symbols and LZW complexity, approximating Kolmogorov complexity. It also computes distances between sequences based on LZW complexity, enabling similarity analysis for symbolic time series.

**Example**:
```python
from slearn import lzw_string_generator, lzw_string_seeds

# Generate a single string with 2 symbols and target complexity 3
str_, str_complex = lzw_string_generator(2, 3, priorise_complexity=True, random_state=2)
print(f"string: {str_}, complexity: {str_complex}")

# Same, but prioritize symbol count over complexity
str_, str_complex = lzw_string_generator(2, 3, priorise_complexity=False, random_state=2)
print(f"string: {str_}, complexity: {str_complex}")

# Generate a library of strings with varying symbols and complexities
df_strings = lzw_string_seeds(symbols=[2, 3], complexity=[3, 6, 7], priorise_complexity=False, random_state=0)
print(df_strings)
```

**Output**:
```
string: BAA, complexity: 3
string: BAB, complexity: 3
   nr_symbols  LZW_complexity  length       string
0           2               3       3          ABA
1           2               6       8     BABBABBA
2           2               7      11  BAAABABAAAA
3           3               3       3          BAC
4           3               6       6       ABCACB
5           3               7       8     ABCAAABB
```

### 2. Benchmarking Deep Learning Models

`slearn` provides tools to benchmark the memorization capabilities of deep learning models (e.g., LSTMs, GRUs, Transformers) on symbolic sequences. The `benchmark_models` function generates performance reports and visualizations.

**Example**:
```python
from slearn.deep_models import LSTMModel, GRUModel, TransformerModel, GPTLikeModel
from slearn.simulation import benchmark_models

model_list = [LSTMModel, GRUModel, TransformerModel, GPTLikeModel]
benchmark_models(
    model_list,
    symbols_list=[2, 4, 6, 8],          # Number of unique symbols
    complexities=[210, 230, 250, 270, 290],  # Target LZW complexities
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

Custom models can be implemented following the examples in [slearn/deep_models.py](https://github.com/nla-group/slearn/blob/master/slearn/deep_models.py).

### 3. Symbolic Time Series Representations

`slearn` supports multiple Symbolic Aggregate Approximation (SAX) variants and the ABBA method for time series symbolization. The following table summarizes the implemented methods:

| Algorithm | Time Series Type | Segmentation | Features Extracted | Symbolization | Reconstruction |
|-----------|------------------|--------------|--------------------|---------------|----------------|
| **SAX**   | Univariate       | Fixed-size segments | Mean (PAA) | Gaussian breakpoints, single symbol per segment | Piecewise constant from PAA values |
| **SAX-TD**| Univariate       | Fixed-size segments | Mean (PAA), slope | Mean to symbol, trend suffix ('u', 'd', 'f') | Linear trends from PAA and slopes |
| **eSAX**  | Univariate       | Fixed-size segments | Min, mean, max | Three symbols per segment (min, mean, max) | Quadratic interpolation from min, mean, max |
| **mSAX**  | Multivariate     | Fixed-size segments | Mean per dimension | One symbol per dimension per segment | Piecewise constant per dimension |
| **aSAX**  | Univariate       | Adaptive segments (local variance) | Mean (PAA) | Gaussian breakpoints, single symbol per segment | Piecewise constant from adaptive segments |
| **ABBA**  | Univariate       | Adaptive piecewise linear segments | Length, increment | Clustering (k-means), symbols assigned to clusters | Piecewise linear from cluster centers |

**Example**:
```python
import numpy as np
from slearn.symbols import SAX, SAXTD, ESAX, MSAX, ASAX

def test_sax_variant(model, ts, t, name, is_multivariate=False):
    symbols = model.fit_transform(ts)
    recon = model.inverse_transform()
    print(f"{name} reconstructed length: {len(recon)}")
    return np.sqrt(np.mean((ts - recon) ** 2))  # RMSE

# Generate test time series
np.random.seed(42)
t = np.linspace(0, 10, 100)
ts = np.sin(t) + np.random.normal(0, 0.1, 100)  # Univariate
ts_multi = np.vstack([np.sin(t), np.cos(t)]).T + np.random.normal(0, 0.1, (100, 2))  # Multivariate

# Test SAX variants
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

### 4. String Distance and Similarity Metrics

`slearn` provides interfaces for computing string distances and similarities, including normalized versions, based on formal definitions.

**Example**:
```python
from slearn.dmetric import (
    damerau_levenshtein_distance,
    jaro_winkler_distance,
    normalized_damerau_levenshtein_distance,
    normalized_jaro_winkler_distance
)

print(damerau_levenshtein_distance("cat", "act"))  # Output: 1
print(jaro_winkler_distance("martha", "marhta"))   # Output: 0.961
print(normalized_damerau_levenshtein_distance("cat", "act"))  # Output: 0.333
print(normalized_jaro_winkler_distance("martha", "marhta"))   # Output: 0.961
```

## Supported Classifiers

`slearn` integrates with scikit-learn classifiers for symbolic sequence analysis: [TO DO]


## Documentation

Comprehensive documentation is available at [slearn.readthedocs.io](https://slearn.readthedocs.io/en/latest/).

## Citation

If you use `slearn` or the `LZWStringLibrary` in your research, please cite:

```bibtex
@InProceedings{10.1007/978-3-031-37963-5_53,
    author="Cahuantzi, Roberto and Chen, Xinye and Güttel, Stefan",
    title="A Comparison of LSTM and GRU Networks for Learning Symbolic Sequences",
    booktitle="Intelligent Computing",
    year="2023",
    publisher="Springer Nature Switzerland",
    pages="771--785"
}
```

For questions or issues, contact the maintainers via email.

## License

This project is licensed under the [MIT License](https://github.com/nla-group/slearn/blob/master/LICENSE).

## Contributing

Contributions to `slearn` are welcome! To contribute:
1. Fork the repository: [github.com/nla-group/slearn](https://github.com/nla-group/slearn).
2. Create a branch for your feature or bug fix.
3. Submit a pull request with a clear description of changes.
4. Ensure tests pass (see `unittests.yml` workflow).

**TODO List**:
- Add language modeling functionalities.
- Expand and refine documentation.
- Optimize performance for large-scale sequence generation and processing.
