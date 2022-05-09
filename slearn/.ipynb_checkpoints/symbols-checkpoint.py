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

import warnings
import collections
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans


def symbolsAssign(clusters):
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
    
    alphabet = ['A','a','B','b','C','c','D','d','E','e',
                'F','f','G','g','H','h','I','i','J','j',
                'K','k','L','l','M','m','N','n','O','o',
                'P','p','Q','q','R','r','S','s','T','t',
                'U','u','V','v','W','w','X','x','Y','y','Z','z']
                
    clusters = pd.Series(clusters)
    N = len(clusters.unique())

    cluster_sort = [0] * N 
    counter = collections.Counter(clusters)
    for ind, el in enumerate(counter.most_common()):
        cluster_sort[ind] = el[0]

    if N >= len(alphabet):
        alphabet = [chr(i+33) for i in range(0, N)]
    else:
        alphabet = alphabet[:N]
    hashm = dict(zip(cluster_sort + alphabet, alphabet + cluster_sort))
    strings = [hashm[i] for i in clusters]
    return strings, hashm



class SAX:
    """Modified from https://github.com/nla-group/TARZAN"""
   
    def __init__(self, *, width = 2, n_paa_segments=None, k = 5, return_list=False, verbose=True):
        if n_paa_segments is not None:
            # if verbose == True:
                # warnings.warn("Set width to ``len(ts) // n_paa_segments''")
            self.n_paa_segments = n_paa_segments
            self.width = None 
        else:
            self.n_paa_segments = None
            self.width = width
        self.number_of_symbols = k
        self.return_list = return_list
        self.mu, self.std = 0, 1
        
        
    def transform(self, time_series):
        if self.width is None:
            self.width = len(time_series) // self.n_paa_segments
        self.mu = np.mean(time_series)
        self.std = np.std(time_series)
        if self.std == 0:
            self.std = 1
        time_series = (time_series - self.mu)/self.std
        compressed_time_series = self.paa_mean(time_series)
        symbolic_time_series = self._digitize(compressed_time_series)
        return symbolic_time_series

    
    def inverse_transform(self, symbolic_time_series):
        compressed_time_series = self._reverse_digitize(symbolic_time_series)
        time_series = self._reconstruct(compressed_time_series)
        time_series = time_series*self.std + self.mu
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
        strings, self.hashm = symbolsAssign(labels)
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
            j = self.hashm[s]
            ts.append(breakpoint_values[j])
        return ts
    
    

    
# python implementation for aggregation
def aggregate(data, sorting="2-norm", tol=0.5): # , verbose=1
    """aggregate the data
    Parameters
    ----------
    data : numpy.ndarray
        the input that is array-like of shape (n_samples,).
    sorting : str
        the sorting method for aggregation, default='2-norm', alternative option: '1-norm' and 'lexi'.
    tol : float
        the tolerance to control the aggregation. if the distance between the starting point 
        of a group and another data point is less than or equal to the tolerance,
        the point is allocated to that group.  
    Returns
    -------
    labels (numpy.ndarray) : 
        the group categories of the data after aggregation
    
    splist (list) : 
        the list of the starting points
    
    nr_dist (int) :
        number of pairwise distance calculations
    """

    splist = list() # store the starting points
    len_ind = data.shape[0]
    if sorting == "2-norm": 
        sort_vals = np.linalg.norm(data, ord=2, axis=1)
        ind = np.argsort(sort_vals)
    elif sorting == "1-norm": 
        sort_vals = np.linalg.norm(data, ord=1, axis=1)
        ind = np.argsort(sort_vals)
    else:
        ind = np.lexsort((data[:,1], data[:,0]), axis=0) 
        
    lab = 0
    labels = [-1]*len_ind
    nr_dist = 0 
    
    for i in range(len_ind): # tqdm（range(len_ind), disable=not verbose)
        sp = ind[i] # starting point
        if labels[sp] >= 0:
            continue
        else:
            clustc = data[sp,:] 
            labels[sp] = lab
            num_group = 1

        for j in ind[i:]:
            if labels[j] >= 0:
                continue

            dat = clustc - data[j,:]
            dist = np.inner(dat, dat)
            nr_dist += 1
                
            if dist <= tol**2:
                num_group += 1
                labels[j] = lab
            else: # apply early stopping
                if sorting == "2-norm" or sorting == "1-norm":
                    if (sort_vals[j] - sort_vals[sp] > tol):
                        break       
                else:
                    if ((data[j,0] - data[sp,0] == tol) and (data[j,1] > data[sp,1])) or (data[j,0] - data[sp,0] > tol): 
                        break
        splist.append([sp, lab] + [num_group] + list(data[sp,:]) ) # respectively store starting point
                                                               # index, label, number of neighbor objects, center (starting point).
        lab += 1

    return np.array(labels), splist, nr_dist 



def compress(ts, tol=0.5, max_len=np.inf):
    """
    Approximate a time series using a continuous piecewise linear function.
    Parameters
    ----------
    ts - numpy ndarray
        Time series as input of numpy array
    Returns
    -------
    pieces - numpy array
        Numpy ndarray with three columns, each row contains length, increment, error for the segment.
    """

    start = 0
    end = 1
    pieces = list() # np.empty([0, 3])
    x = np.arange(0, len(ts))
    epsilon =  np.finfo(float).eps

    while end < len(ts):
        inc = ts[end] - ts[start]
        err = ts[start] + (inc/(end-start))*x[0:end-start+1] - ts[start:end+1]
        err = np.inner(err, err)

        if (err <= tol*(end-start-1) + epsilon) and (end-start-1 < max_len):
            (lastinc, lasterr) = (inc, err) 
            end += 1
        else:
            # pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
            pieces.append([end-start-1, lastinc, lasterr])
            start = end - 1

    # pieces = np.vstack([pieces, np.array([end-start-1, lastinc, lasterr])])
    pieces.append([end-start-1, lastinc, lasterr])

    return pieces




class fABBA:
    def __init__ (self, tol=0.1, alpha=0.5,  sorting='2-norm', scl=1, verbose=1, max_len=np.inf):
        """
        
        Parameters
        ----------
        tol - float
            Control tolerence for compression, default as 0.1.
        scl - int
            Scale for length, default as 1, means 2d-digitization, otherwise implement 1d-digitization.
        verbose - int
            Control logs print, default as 1, print logs.
        max_len - int
            The max length for each segment, default as np.inf. 
        
        """
        
        self.tol = tol
        self.scl = scl
        self.verbose = verbose
        self.max_len = max_len
        self.alpha = alpha
        self.sorting = sorting
        self.compression_rate = None
        self.digitization_rate = None
        
    

    def fit_transform(self, series):
        """ 
        Compress and digitize the time series together.
        
        Parameters
        ----------
        series - array or list
            Time series.

        alpha - float
            Control tolerence for digitization, default as 0.5.

        string_form - boolean
            Whether to return with string form, default as True.
        """
        series = np.array(series).astype(np.float64)
        pieces = np.array(compress(ts=series, tol=self.tol, max_len=self.max_len))
        strings = self.digitize(pieces[:,0:2])
        self.compression_rate = pieces.shape[0] / series.shape[0]
        self.digitization_rate = self.centers.shape[0] / pieces.shape[0]
        if self.verbose in [1, 2]:
            print("""Compression: Reduced series of length {0} to {1} segments.""".format(series.shape[0], pieces.shape[0]),
                """Digitization: Reduced {} pieces""".format(len(strings)), "to", self.centers.shape[0], "symbols.")  
        # strings = ''.join(strings)
        return strings
    
    

    def digitize(self, pieces, early_stopping=True):
        """
        Greedy 2D clustering of pieces (a Nx2 numpy array),
        using tolernce tol and len/inc scaling parameter scl.
        In this variant, a 'temporary' cluster center is used 
        when assigning pieces to clusters. This temporary cluster
        is the first piece available after appropriate scaling 
        and sorting of all pieces. It is *not* necessarily the 
        mean of all pieces in that cluster and hence the final
        cluster centers, which are just the means, might achieve 
        a smaller within-cluster tol.
        """

        if self.sorting not in ['2-norm', '1-norm', 'lexi']:
            raise ValueError("Please refer to a specific and correct sorting way, namely '2-norm', '1-norm' and 'lexi'")
        
        _std = np.std(pieces, axis=0) # prevent zero-division
        if _std[0] == 0:
             _std[0] = 1
        if _std[1] == 0:
             _std[1] = 1
                
        npieces = pieces * np.array([self.scl, 1]) / _std
        labels, self.splist, self.nr_dist = aggregate(npieces, self.sorting, self.alpha) # other two variables are used for experiment
        centers = np.zeros((0,2))
        for c in range(len(self.splist)):
            indc = np.argwhere(labels==c)
            center = np.mean(pieces[indc,:], axis=0)
            centers = np.r_[ centers, center ]
        self.centers = centers
        strings, self.hashmap = symbolsAssign(labels)
        return strings
    

    
    def inverse_transform(self, strings, start=0):
        pieces = self.inverse_digitize(strings, self.centers, self.hashmap)
        pieces = self.quantize(pieces)
        time_series = self.inverse_compress(pieces, start)
        return time_series

    

    def inverse_digitize(self, strings, centers, hashmap):
        pieces = np.empty([0,2])
        for p in strings:
            pc = centers[int(hashmap[p])]
            pieces = np.vstack([pieces, pc])
        return pieces[:,0:2]


    
    def quantize(self, pieces):
        if len(pieces) == 1:
            pieces[0,0] = round(pieces[0,0])
        else:
            for p in range(len(pieces)-1):
                corr = round(pieces[p,0]) - pieces[p,0]
                pieces[p,0] = round(pieces[p,0] + corr)
                pieces[p+1,0] = pieces[p+1,0] - corr
                if pieces[p,0] == 0:
                    pieces[p,0] = 1
                    pieces[p+1,0] -= 1
            pieces[-1,0] = round(pieces[-1,0],0)
        return pieces

    

    def inverse_compress(self, pieces, start):
        """Modified from ABBA package, please see ABBA package to see guidance."""
        
        time_series = [start]
        # stitch linear piece onto last
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            #print(x)
            y = time_series[-1] + x
            time_series = time_series + y[1:].tolist()

        return time_series
    
    
    


class ABBA:
    def __init__ (self, tol=0.1, k_cluster=10, verbose=1, max_len=np.inf):
        """
        
        Parameters
        ----------
        tol - float
            Control tolerence for compression, default as 0.1.
        
        k_cluster - int
            Number of symbols used for digitization.
        
        verbose - int
            Control logs print, default as 1, print logs.
        
        max_len - int
            The max length for each segment, default as np.inf. 
        
        """
        
        self.tol = tol
        self.verbose = verbose
        self.max_len = max_len
        self.k_cluster = k_cluster
        self.compression_rate = None
        self.digitization_rate = None
    
        

    def fit_transform(self, series):
        """ 
        Compress and digitize the time series together.
        
        Parameters
        ----------
        series - array or list
            Time series.

        alpha - float
            Control tolerence for digitization, default as 0.5.

        string_form - boolean
            Whether to return with string form, default as True.
        """
        series = np.array(series).astype(np.float64)
        pieces = np.array(compress(ts=series, tol=self.tol, max_len=self.max_len))
        strings = self.digitize(pieces[:,0:2])
        self.compression_rate = pieces.shape[0] / series.shape[0]
        self.digitization_rate = self.centers.shape[0] / pieces.shape[0]
        if self.verbose in [1, 2]:
            print("""Compression: Reduced series of length {0} to {1} segments.""".format(series.shape[0], pieces.shape[0]),
                """Digitization: Reduced {} pieces""".format(len(strings)), "to", self.centers.shape[0], "symbols.")  
        # strings = ''.join(strings)
        return strings
    
    

    def digitize(self, pieces, early_stopping=True):
        """
        Greedy 2D clustering of pieces (a Nx2 numpy array),
        using tolernce tol and len/inc scaling parameter scl.
        In this variant, a 'temporary' cluster center is used 
        when assigning pieces to clusters. This temporary cluster
        is the first piece available after appropriate scaling 
        and sorting of all pieces. It is *not* necessarily the 
        mean of all pieces in that cluster and hence the final
        cluster centers, which are just the means, might achieve 
        a smaller within-cluster tol.
        """

        _std = np.std(pieces, axis=0) # prevent zero-division
        if _std[0] == 0:
             _std[0] = 1
        if _std[1] == 0:
             _std[1] = 1
                
        npieces = pieces / _std
        print(npieces.shape)
        kmeans = KMeans(n_clusters=self.k_cluster, random_state=0).fit(npieces)
        labels = kmeans.labels_
        self.centers = kmeans.cluster_centers_*_std
        strings, self.hashmap = symbolsAssign(labels)
        return strings
    

    
    def inverse_transform(self, strings, start=0):
        pieces = self.inverse_digitize(strings, self.centers, self.hashmap)
        pieces = self.quantize(pieces)
        time_series = self.inverse_compress(pieces, start)
        return time_series

    

    def inverse_digitize(self, strings, centers, hashmap):
        pieces = np.empty([0,2])
        for p in strings:
            pc = centers[int(hashmap[p])]
            pieces = np.vstack([pieces, pc])
        return pieces[:,0:2]


    
    def quantize(self, pieces):
        if len(pieces) == 1:
            pieces[0,0] = round(pieces[0,0])
        else:
            for p in range(len(pieces)-1):
                corr = round(pieces[p,0]) - pieces[p,0]
                pieces[p,0] = round(pieces[p,0] + corr)
                pieces[p+1,0] = pieces[p+1,0] - corr
                if pieces[p,0] == 0:
                    pieces[p,0] = 1
                    pieces[p+1,0] -= 1
            pieces[-1,0] = round(pieces[-1,0],0)
        return pieces

    

    def inverse_compress(self, pieces, start):
        """Modified from ABBA package, please see ABBA package to see guidance."""
        
        time_series = [start]
        # stitch linear piece onto last
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            #print(x)
            y = time_series[-1] + x
            time_series = time_series + y[1:].tolist()

        return time_series
    
    