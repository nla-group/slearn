---
title: 'slearn: The machine learning benchmark for symbolic time series representation'
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
    
  - name: Stefan G\"{u}ttel
    orcid: 0000-0003-1494-4478
    affiliation: 1
affiliations:
  - name: Department of Mathematics, The University of Manchester
    index: 1
date: 13 March 2022
bibliography: paper.bib
---

## Summary

Symbolic representations of time series have proved their usefulness in the field of time series motif discovery \citep{10.1007/s10618-007-0064-z}, clustering \citep{10.1007/s10618-007-0064-z}, classification \citep{Schfer2014TheBI}, forecasting \citep{EG20b}, anomaly detection \citep{10.1145/775047.775128}, etc. Transforming time series into symbolic representation reduces the dimensionality of time series by compressing time series while preserving its underlying structure, and therefore speedups the downstream time series task. With the meaningful patterns underlying informative symbols, a compressed sequence can greatly speed up the algorithm inference time in prediction. Appropriately employing machine learning algorithms on the level of symbols instead of raw time series poses a great challenge to symbolic time series forecasting problems. For the convenience of the research community on studying the symbolic representation, we design a Python module for benchmarking the process of machine learning algorithm practice with symbolic representation, called slearn.  Our software includes some basic symbols preprocessing tools and builds a baseline for various machine learning models through scikit-learn package \cite{scikit-learn} for symbolic prediction. Our library mainly concerns two components, namely string generator, machine learning forecasting. The string generator and compression follow the new research paper \citep{CCG21} while the forecasting follows the \citep{EG20b}. 

## Examples of use

slearn is publicly available on GitHub and can be installed via the pip package manager. The documentation in \url{https://slearn.readthedocs.io/en/latest/?badge=latest} provides detailed functionality guidance and deployment of prediction with SAX \cite{10.1145/882082.882086}, ABBA \cite{EG19b}, and fABBA \cite{CG22a}. slean module offers user-friendly APIs, and consistent parameters for prediction as scikit-learn package. The forecasting empirical result of slearn prediction for amazon stock close price is as shown in \figurename~\ref{demo}.


\begin{figure}[ht]
	\centering
	\includegraphics[width=1\textwidth]{demo1}
	\caption{Prediction with various machine learning models}
	\label{demo}
\end{figure} 


## Statement of need

Currently, there is no software that leverages the symbolic time series representation techniques with a combination of machine learning algorithms for prediction. For the convenience of the machine learning community of developing algorithms on the level of symbolic representation, we develop these tools for benchmark test of symbolic time-series representation and also for practical forecasting use. It can generate large-scale time series data according to the need of the user by specifying the LZW complexity of symbolic sequences, which allows for researchers to develop their machine learning algorithm in this test benchmark. Besides, slean offers fundamental string processing tools including one-hot encoding and Lempel–Ziv–Welch technique \cite{Welch1984}. For the purpose of easy-to-use, the software keeps consistent and simple APIs for employment. New functionality will be added in the future.


## Acknowledgements

Roberto Cahuantzi's work has been funded by the UK's Alan Turing Institute. Stefan Güttel has been supported by a Fellowship of the Alan Turing Institute, EPSRC grant EP/N510129/1.


## References
