# Copyright (c) 2021, nla group, manchester
# All rights reserved. 

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import re
import copy
import random
import warnings
import collections
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm

from sklearn.svm import SVC
# from deepforest import CascadeForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


import warnings


class symbolicML:
    """
    Parameters
    ----------
    classifier_name - str, default=MLPClassifier, 
                      optional choices = {"KNeighborsClassifier", "GaussianProcessClassifier"
                      "QuadraticDiscriminantAnalysis", "DecisionTreeClassifier",
                      "LogisticRegression", "AdaBoostClassifier",
                      "RandomForestClassifier", "GaussianNB",
                      "DeepForest", "LGBM",
                      "SVC", "RBF"}: 
        The classifier you specify for symbols prediction.

    gap - int, default=3:
        The windows size for symbols to be the features, i.e, the dimensions of features.

    random_seed - int, default=0:
        The random state fixed for classifers in scikit-learn.

    verbose - int, default=0:
        log print. Whether to print progress or other messages to stdout.

    Examples
    ----------
    >>> from slearn import symbolicML
    >>> string = 'aaaabbbccd'
    >>> sbml = symbolicML(classifier_name="MLPClassifier", gap=5, random_seed=0, verbose=0)
    >>> x, y = sbml._encoding(string)
    >>> pred = sbml.forecasting(x, y, step=5, hidden_layer_sizes=(3,3), activation='relu', learning_rate_init=0.01)
    >>> print(pred)
    ['b', 'b', 'b', 'c', 'c']
    
    or you can use a more general method like:
    
    >>> from slearn import symbolicML
    >>> string = 'aaaabbbccd'
    >>> string = 'aaaabbbccd'
    >>> sbml = symbolicML(classifier_name="mlp", gap=5, random_seed=0, verbose=0)
    >>> x, y = sbml._encoding(string)
    >>> params = {'hidden_layer_sizes':(10,10), 'activation':'relu', 'learning_rate_init':0.1}
    >>> pred = sbml.forecasting(x, y, step=5, **params)
    >>> print(pred)
    ['b', 'b', 'c', 'c', 'd']
    """
        
    def __init__(self, classifier_name='MLPClassifier', gap=3, random_seed=0, verbose=0):
        self.classifier_name = classifier_name 
        self.init_classifier()
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        self.gap = gap
        self.verbose = verbose
        self.mu = 0
        self.scl = 1
        
        
    def _encoding(self, string):
        if not isinstance(string, list):
            string_split = [s for s in re.split('', string) if s != '']
        else:
            string_split = copy.deepcopy(string)
        self.hashm = dict(zip(set(string_split), np.arange(len(set(string_split)))))
        string_encoding = [self.hashm[i] for i in string_split] 
        if self.gap > len(string_encoding):
            warnings.warn("gap is larger than the series, please reset the gap.")
            self.gap = len(string_encoding) - 1
        x, y = self.construct_train(string_encoding, gap=self.gap)
        return x, y
        
    
    def construct_train(self, series, gap):
        features = list()
        targets = list()
        for i in range(len(series) - gap):
            features.append(series[i:i+gap])
            targets.append(series[i+gap])
        return np.array(features), np.array(targets)


    def init_classifier(self):
        if self.classifier_name == "KNeighborsClassifier":
            self.Classifiers = KNeighborsClassifier
        
        elif self.classifier_name == "GaussianProcessClassifier":
            self.Classifiers = GaussianProcessClassifier
        
        elif self.classifier_name == "QuadraticDiscriminantAnalysis":
            self.Classifiers =  QuadraticDiscriminantAnalysis
        
        elif self.classifier_name == "DecisionTreeClassifier":
            self.Classifiers = DecisionTreeClassifier
        
        elif self.classifier_name == "LogisticRegression":
            self.Classifiers = LogisticRegression
        
        elif self.classifier_name == "AdaBoostClassifier":
            self.Classifiers = AdaBoostClassifier
        
        elif self.classifier_name == "RandomForestClassifier":
            self.Classifiers = AdaBoostClassifier
        
        elif self.classifier_name == "GaussianNB":
            self.Classifiers = GaussianNB
        
        # elif self.classifier_name == "DeepForest":
        #    self.Classifiers = CascadeForestClassifier
        
        elif self.classifier_name == "LGBM":
            # lgb_params = {'boosting_type': 'gbdt',
            #          'learning_rate': 0.5,
            #          'max_depth': 5
            #          }
            self.Classifiers = lgb.LGBMClassifier
        
        elif self.classifier_name == "SVC":
            self.Classifiers = SVC
        
        elif self.classifier_name == "RBF":
            self.Classifiers = RBF

        else: # "MLPClassifier"
            self.Classifiers = MLPClassifier
         
    def forecasting(self, x, y, step=5, inversehash=None, centers=None, **params):
        try:
            cparams = copy.deepcopy(params)
            if "verbose" in self.Classifiers().__dict__:
                if not self.verbose:
                    cparams['verbose'] = 0
            if "random_state" in self.Classifiers().__dict__:
                cparams['random_state'] = 0
            clf = self.Classifiers(**cparams)
            clf.fit(x, y)
        except:
            print("fail to set_random_state.")
            clf = self.Classifiers(**params)
            clf.fit(x, y)


        if inversehash == None:
            for i in range(step):
                last_x = np.hstack((x[-1][1:], y[-1]))
                pred = clf.predict(np.expand_dims(last_x, axis=0))
                x = np.vstack((x, last_x))
                y = np.hstack((y, pred))
            inversehash = dict(zip(self.hashm.values(), self.hashm.keys()))
            symbols_pred = [inversehash[n] for n in y[-step:]]
        else:
            for i in range(step):
                last_x = np.hstack((x[-1][1:], (centers[y[-1]] - self.mu)/self.scl))
                pred = clf.predict(np.expand_dims(last_x, axis=0))
                x = np.vstack((x, last_x))
                y = np.hstack((y, pred))
            symbols_pred = [inversehash[n] for n in y[-step:]]
        return symbols_pred


    
    
class SAX:
    """Modified from https://github.com/nla-group/TARZAN"""
    def __init__(self, *, w = 2, n_paa_segments=None, k = 5, return_list=False, verbose=True):
        if n_paa_segments is not None:
            if verbose == True:
                warnings.warn("Deprecated parameter w. Set width to ``len(ts) // n_paa_segments''")
            self.width = len(ts) // n_paa_segments
        else:
            self.width = w
        self.number_of_symbols = k
        self.return_list = return_list

    def transform(self, time_series):
        compressed_time_series = self.paa_mean(time_series)
        symbolic_time_series = self._digitize(compressed_time_series)
        return symbolic_time_series

    def inverse_transform(self, symbolic_time_series):
        compressed_time_series = self._reverse_digitize(symbolic_time_series)
        time_series = self._reconstruct(compressed_time_series)
        return time_series

    def paa_mean(self, ts):
        if len(ts) % self.width != 0:
            warnings.warn("Result truncates, width does not divide length")
        return [np.mean(ts[i*self.width:np.min([len(ts), (i+1)*self.width])]) for i in range(int(np.floor(len(ts)/self.width)))]

    def _digitize(self, ts):
        symbolic_ts = self._gaussian_breakpoints(ts)
        return symbolic_ts

    def _gaussian_breakpoints(self, ts):
        # Construct Breakpoints
        breakpoints = np.hstack(
            (norm.ppf([float(a) / self.number_of_symbols for a in range(1, self.number_of_symbols)], scale=1), 
             np.inf))
        labels = []
        for i in ts:
            for j in range(len(breakpoints)):
                if i < breakpoints[j]:
                    labels.append(j)
                    break
        strings, self.hashm, self.inverse_hashm = self.symbolsAssign(labels)
        if not self.return_list:
            strings = "".join(strings)
        return strings

    def _reconstruct(self, reduced_ts):
        return self._reverse_pca(reduced_ts)
    
    def _reverse_pca(self, ts):
        return np.kron(ts, np.ones([1,self.width])[0])

    def _reverse_digitize(self, symbolic_ts):
        return self._reverse_gaussian_breakpoints(symbolic_ts)
    
    def _reverse_gaussian_breakpoints(self, symbols):
        breakpoint_values = norm.ppf([float(a) / (2 * self.number_of_symbols) for a in range(1, 2 * self.number_of_symbols, 2)], scale=1)
        ts = []
        for s in symbols:
            j = self.inverse_hashm[s]
            ts.append(breakpoint_values[j])
        return ts
    
    def symbolsAssign(self, clusters):
        """ automatically assign symbols to different clusters, start with '!'
            from https://github.com/nla-group/fABBA.
        
        Parameters
        ----------
        clusters - list or pd.Series or array
                the list of clusters.
        ----------
        Return:
        
        symbols(list of string), inverse_hash(dict): repectively for the
        corresponding symbolic sequence and the hashmap for inverse transform.
        
        """
        
        clusters = pd.Series(clusters)
        N = len(clusters.unique())

        cluster_sort = [0] * N 
        counter = collections.Counter(clusters)
        for ind, el in enumerate(counter.most_common()):
            cluster_sort[ind] = el[0]

        alphabet= [chr(i) for i in range(33,33 + N)]
        hashm = dict(zip(cluster_sort, alphabet))
        inverse_hashm = dict(zip(alphabet, cluster_sort))
        strings = [hashm[i] for i in clusters]
        return strings, hashm, inverse_hashm
    
    
    
    
class slearn(symbolicML):
    """
    Parameters
    ----------
    series - numpy.ndarray:
        The numeric time series. 
        
    method - str {'ABBA', 'SAX'}:
        The symbolic time series representation.
        We use fABBA for ABBA method.
    
    classifier_name - str, default=MLPClassifier, 
                      optional choices = {"KNeighborsClassifier", "GaussianProcessClassifier"
                      "QuadraticDiscriminantAnalysis", "DecisionTreeClassifier",
                      "LogisticRegression", "AdaBoostClassifier",
                      "RandomForestClassifier", "GaussianNB",
                      "LGBM", "SVC", "RBF"}: 
        The classifier you specify for symbols prediction.

    gap - int, default=3:
        The windows size for symbols to be the features, i.e, the dimensions of features.
    
    step - int, default=1,
        The number of symbols for prediction.
        
    random_seed - int, default=0:
        The random state fixed for classifers in scikit-learn.

    verbose - int, default=0:
        log print. Whether to print progress or other messages to stdout.
    

    Examples
    ----------
    Use Multi-layer Perceptron classifier for prediction:
    >>> import numpy as np
    >>> from slearn import slearn
    >>> ts = [np.sin(0.05*i) for i in range(1000)]
    >>> sl = slearn(series=ts, method='ABBA', gap=10, step=1, tol=0.5, alpha=0.5, form='numeric', classifier_name="MLPClassifier", random_seed=1, verbose=0)
    >>> sl.predict(hidden_layer_sizes=(10,10), activation='relu', learning_rate_init=0.1)
    array([-0.34127142, -0.37226769, -0.40326396, -0.43426023, -0.4652565 ,
       -0.49625277, -0.52724904, -0.55824531, -0.58924158, -0.62023785,
       -0.65123412, -0.68223039, -0.71322666, -0.74422293, -0.7752192 ,
       -0.80621547, -0.83721174, -0.86820801, -0.89920428, -0.93020055,
       -0.96119682, -0.99219309, -1.02318936, -1.05418563, -1.0851819 ,
       -1.11617817, -1.14717444, -1.17817071, -1.20916698, -1.24016325,
       -1.27115952, -1.30215579, -1.33315206, -1.36414833, -1.3951446 ,
       -1.42614087, -1.45713714, -1.48813341, -1.51912968, -1.55012595,
       -1.58112222, -1.61211849, -1.64311476, -1.67411103, -1.7051073 ,
       -1.73610357, -1.76709984, -1.79809611, -1.82909238, -1.86008865,
       -1.89108492, -1.92208119, -1.95307746, -1.98407373, -2.01507   ,
       -2.04606627, -2.07706254, -2.10805881, -2.13905508, -2.17005135,
       -2.20104762, -2.23204389, -2.26304016])
    >>> sl.predict(step=10, form='string', hidden_layer_sizes=(3,3), activation='relu', learning_rate_init=0.1)
    ['"', '!', '"', '!', '"', '!', '"', '!', '"', '!']
    
    or you can use a more general method like:
    
    >>> import numpy as np
    >>> from slearn import slearn
    >>> ts = [np.sin(0.05*i) for i in range(1000)]
    >>> sl = slearn(series=ts, method='ABBA', gap=10, step=1, tol=0.5,  alpha=0.5, form='numeric', classifier_name="MLPClassifier", random_seed=1, verbose=0)
    >>> params = {'hidden_layer_sizes':(10,10), 'activation':'relu', 'learning_rate_init':0.1}
    >>> sl.predict(**params)
    array([-0.34127142, -0.37226769, -0.40326396, -0.43426023, -0.4652565 ,
           -0.49625277, -0.52724904, -0.55824531, -0.58924158, -0.62023785,
           -0.65123412, -0.68223039, -0.71322666, -0.74422293, -0.7752192 ,
           -0.80621547, -0.83721174, -0.86820801, -0.89920428, -0.93020055,
           -0.96119682, -0.99219309, -1.02318936, -1.05418563, -1.0851819 ,
           -1.11617817, -1.14717444, -1.17817071, -1.20916698, -1.24016325,
           -1.27115952, -1.30215579, -1.33315206, -1.36414833, -1.3951446 ,
           -1.42614087, -1.45713714, -1.48813341, -1.51912968, -1.55012595,
           -1.58112222, -1.61211849, -1.64311476, -1.67411103, -1.7051073 ,
           -1.73610357, -1.76709984, -1.79809611, -1.82909238, -1.86008865,
           -1.89108492, -1.92208119, -1.95307746, -1.98407373, -2.01507   ,
           -2.04606627, -2.07706254, -2.10805881, -2.13905508, -2.17005135,
           -2.20104762, -2.23204389, -2.26304016])
           
    Use Gaussian Naive Bayes classifier for prediction:
    >>> import numpy as np
    >>> ts = [np.sin(0.05*i) for i in range(1000)]
    >>> sl = slearn(series=ts, method='fABBA', gap=10, step=1, tol=0.5,  alpha=0.5, form='numeric', classifier_name="GaussianNB", random_seed=1, verbose=0)
    >>> params = {'var_smoothing':0.001}
    >>> sl.predict(**params)
    """
        
    def __init__(self, series, method='ABBA', gap=5, step=10,
                 classifier_name="MLPClassifier",
                 form='numeric', random_seed=0, verbose=0, **params):
        
        self.method = method
        self.random_seed = random_seed
        self.gap = gap
        self.classifier_name=classifier_name
        self.verbose = verbose
        self.form = form
        self.step = step
        if not isinstance(series, np.ndarray):
            series = np.array(series)
            
        self.mu = series.mean()
        self.scl = series.std()
        
        if self.scl == 0:
            self.scl = 1
        scale_series = (series - self.mu) / self.scl
        
        self.start = scale_series[0]
        self.length = len(series)
        self.params_secure()
        
        if self.method == 'ABBA':
            from fABBA import fabba_model
            try:
                self.s_model = fabba_model(tol=params['tol'], alpha=params['alpha'], 
                                           sorting=params['sorting'], verbose=self.verbose, 
                                           return_list=True)
            except:
                self.s_model = fabba_model(tol=0.1, alpha=0.1, sorting='2-norm', verbose=self.verbose,
                                          return_list=True)
            
            self.string = self.s_model.fit_transform(scale_series)         
            
        elif self.method == 'SAX':
            width = self.length // params['n_paa_segments']
            self.s_model = SAX(w=width, k=params['k'], return_list=True)
            self.string = self.s_model.transform(scale_series)
            
        else:
            warnings.warn(
                "Sorry, there is no {} method for now. Will use the 'ABBA' method".format(self.method))
            self.s_model = fabba_model(tol=0.1, alpha=0.1, sorting='2-norm', 
                                       verbose=self.verbose, return_list=True)
                
        if self.gap >= len(self.string):
            warnings.warn("Parameters are not appropriate, classifier might not converge.")
            warnings.warn("Degenerate to trivial case that self.gap=1.")
            self.gap = 1
            
            
    def predict(self, **params):
        self.cmodel = symbolicML(classifier_name=self.classifier_name,
                          gap=self.gap, 
                          random_seed=self.random_seed, 
                          verbose=self.verbose
                         )
        if self.verbose:
            print("-------- Config --------")
            print("The length of time series: ", self.length)
            print("The number of symbols: ", len(self.string))
            print("The dimension of features is: ", self.gap)
            print("The number of symbols to be predicted: ", self.step)
        
        x, y = self.cmodel._encoding(self.string)
        if self.form == 'string':
            return self.cmodel.forecasting(x, y, step=self.step, **params)
        else:
            pred = self.cmodel.forecasting(x, y, step=self.step, **params)
            if self.method == 'ABBA':
                inverse_ts = self.s_model.inverse_transform(self.string+pred, self.start)
            else:
                inverse_ts = self.s_model.inverse_transform(self.string+pred)
            inverse_ts = np.array(inverse_ts) * self.scl + self.mu
            return inverse_ts[self.length:]
    
    
    def params_secure(self):
        if not isinstance(self.method, str):
            raise ValueError("Please ensure method is string type!")
        if not (isinstance(self.random_seed, float) or isinstance(self.random_seed, int)):
            raise ValueError("Please ensure random_seed is numeric type!")
        if (not isinstance(self.gap, int)) and self.gap > 0:
            raise ValueError("Please ensure gap is integer!")
        if (not isinstance(self.step, int)) and self.step > 0:
            raise ValueError("Please ensure gap is integer!")
        if not isinstance(self.classifier_name, str):
            raise ValueError("Please ensure classifier_name is string type!")
            
            
