---
title: 'slearn: A symbolic time series test tool'
tags:
  - machine learning
  - time series
  - symbolic representation
  - data mining
authors:
  - name: Roberto Cahuantzi
    orcid: 0000-0002-0212-6825
    affiliation: 1
    
  - name: Xinye Chen
    orcid: 0000-0003-1778-393X
    affiliation: 1
    
  - name: Stefan Güttel
    orcid: 0000-0003-1494-4478
    affiliation: 1
affiliations:
  - name: Department of Mathematics, The University of Manchester
    index: 1
date: 13 March 2022
bibliography: paper.bib
---

## Summary

Symbolic representations of time series have proved their usefulness in the field of motif discovery [@10.1007/s10618-007-0064-z}, clustering [@10.1007/s10618-007-0064-z], classification [@Schfer2014TheBI], forecasting [@EG20b], anomaly detection [@10.1145/775047.775128], etc. Transformations of time series into symbolic form often aim to reduce the data dimensionality while preserving the essential time series features (like for example peaks). Furthermore, symbolic representations can help reduce data noise and speedup downstream tasks in machine learning with time series. Symbolic representations essentially turn regression tasks on time series into classification problems, which can lead to reduced training time and reduced sensitivity to hyperparameters [@EG20b]. For the convenience of the research community we present a Python module for benchmarking machine learning algorithms with symbolic representations, called `slearn`.  Our software includes basic symbolic preprocessing tools and provides a baseline for various machine learning models through scikit-learn package [@scikit-learn]. The module comprises two main components, a string generator and machine learning-based forecasting tools. The string generator follows the new research paper [@CCG21} and the forecasting tools follow [@EG20b]. 

## Examples of use

`slearn` is publicly available on GitHub and can be installed via the pip package manager. The documentation in \url{https://slearn.readthedocs.io/en/latest/?badge=latest} provides detailed functionality guidance and deployment of prediction using SAX [@10.1145/882082.882086], ABBA [@EG19b], and fABBA [@CG22a]. `slearn` offers user-friendly APIs and consistent parameters inspired by the scikit-learn package. Figure \autoref{fig:demo} shows a forecast of the Amazon stock close price produced by `slearn`.

![Prediction with various machine learning models.\label{fig:demo}](demo1.png)

## Statement of need

Currently, there is no software that leverages the symbolic time series representation techniques with a combination of machine learning algorithms for prediction. For the convenience of the machine learning community of developing algorithms on the level of symbolic representation, we develop these tools for benchmark test of symbolic time-series representation and also for practical forecasting use. It can generate large-scale time series data according to the need of the user by specifying the Lempel–Ziv–Welch compression (LZW) [@Welch1984] for symbolic sequences, which allows researchers to develop their machine learning algorithm in this test benchmark. Besides, slean offers fundamental string processing tools including one-hot encoding and LZW compression. For the purpose of easy-to-use, the software keeps consistent and simple APIs for employment. New functionality will be added in the future.


## Acknowledgements

Roberto Cahuantzi's work has been funded by the UK's Alan Turing Institute. Stefan Güttel has been supported by a Fellowship of the Alan Turing Institute, EPSRC grant EP/N510129/1.


## References
