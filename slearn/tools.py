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


# https://github.com/robcah/RNNExploration4SymbolicTS
from .symbols import *
from random import randint
import pandas as pd
import numpy as np
from itertools import product
import warnings




def Symbols(n=52):
    '''Creation of a dictionary with the compression symbols, restricted
    up to 52 alphabetical characters from 'A' to 'z'
    
    Parameters
    ----------
    n : int
        Number of symbols within the dictionary.
        
    Return
    ------
    dict :
        Dictionary containing the symbols an their numerical code.
    '''

    range_ = 123
    collection = [chr(i) for i in np.arange(range_) if chr(i).isalpha()]
    dict_symbols = {symbol:i for i,symbol in enumerate(collection) if i<n}
    
    return dict_symbols



def LZWcompress(uncompressed):
    """LZW compress a string to a list of numerical codes, it is 
    restricted to alphabetical values from 'A' to 'z'.
    Based on from https://rosettacode.org/wiki/LZW_compression#Python

    Parameters
    ----------
    uncompressed : string
        String to be compressed using Lempel-Ziv-Welch method, as explained in 
        'A Technique for High-Performance Data Compression', Welch, 1984.

    Returns
    -------
    list of integers
        Each element in the list is a code equivalent to a character or a 
        groups of characters.
    """
 
    # Build the dictionary.
    dictionary = Symbols()
    dict_size = len(dictionary)
 
    w = ''
    result = []
    for c in uncompressed:
        # Concatenates previous symbol with current one in the dictionary
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            # Appends the compression single code to the compression collection
            result.append(dictionary[w])
                            
            # Adds wc to the end of the dictionary
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
 
    # Output the code for w, appending the last symbol.
    if w:
        result.append(dictionary[w])
        
    return result
 
 

def LZWdecompress(compressed):
    """ LZW decompression from a list of integers to a string.

    Parameters
    ----------
    compressed : list of integers
        Each element correspond to a character or combination of characters,
        as explained in 'A Technique for High-Performance Data Compression', 
        Welch, 1984.

    Returns
    -------
    strb :
        The representation of the list of codes into a string.
    """

    # Build the dictionary.
    dictionary = Symbols()
    # Swapping keys and values
    dictionary = {value:key for key, value in dictionary.items()}
    dict_size = len(dictionary)
    
    # Initiation of string building
    string = ''
    w = dictionary[compressed.pop(0)]
    string += w
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        string += entry
 
        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
 
        w = entry
    
    return string 
    
    
    
def Reduce(s):
    """ Reduce a string 's' to its shortest period.

    Parameters
    ----------
    s : str
        This string will be evaluated to find whether it can be reduced to 
        a simpler representation.

    Returns
    -------
    str :
        The result of the reduced string.
    """
    # This loops makes a revision sweeping the string to find coincidences 
    # betwee subdivisions of the string.
    for j in range(1,len(s)//2+1):
        k = j
        while k < len(s):
            if s[k:k+j] != s[0:j]:
                break
            k += j
        if k >= len(s):
            return s[0:j]

    return s



def LZWStringGenerator(nr_symbols, target_complexity, 
                       priorise_complexity=True):
    """Generator of strings based on LZW complexity (elements within the 
        compression symbols' list). If complexity is priorised it will 
        stop the string creation until the aimed complexity is achieved.

    Parameters
    ----------
    nr_symbols : int
        Aimed number of symbols to use to build the string.
    target_complexity : int
        Aimed level of complexity for the string.
    priorise_complexity :  boolean, default True
        If true generates a string with complexity about target_complexity, 
        otherwise the generated string goes above the target complexity until 
        reach the nr_symbols.

    Return
    ------
    str :
        String produced within the specified parameters
    int :
        Level of LZW complexity.
    """
    
    # Caping the highest value of number of symbols to 52.
    if nr_symbols > 52:
        warn0 = 'The strings are limited to alphabetic symbols.'
        warn1 = 'Highest number of symbols is 52.'
        warnings.warn(f'{warn0} {warn1}')
        nr_symbols = 52
    
    # Creation of symbol dictionary
    symbols = Symbols()

    string = 'A'          # current string
    complexity_0 = 1      # current complexity of s
    symbol_max = 1 #65 + 1   # maximal potential symbol
    symbols_pool = ''.join(list(Symbols().keys()))
    stop = True
    
    # Prevents the error of trying to generate a string of complexity >1 with 
    # only 1 symbol.
    if nr_symbols == 1 and target_complexity == 1:
        return 'A', target_complexity
    elif nr_symbols == 1 and target_complexity > 1:
        warn0 = 'Complexities higher than one require more than one symbol.'
        warn1 = 'Output fixed to: ("A",1)'
        warnings.warn(f'{warn0} {warn1}')
        return 'A', 1
    elif nr_symbols > target_complexity:
        return np.nan , 0
    
#     if nr_symbols == target_complexity:
    symbols = [symbols_pool[i] for i in range(nr_symbols)]
    np.random.shuffle(symbols)
    string =  ''.join(symbols)
    complexity_0 = nr_symbols
    symbol_max = nr_symbols
        
    
    # Adds a pseudorandom symbol to the string until either the target 
    # complexity is reached or the number of symbols
    while complexity_0 < target_complexity or not stop:

        # Limits the pool of symbols to select
        symbol_i = randint(0, symbol_max-1)
        string += symbols_pool[symbol_i]
        complexity_0 = len(LZWcompress(Reduce(string)))

        # Switch to create strings without priorising complexity
        if not priorise_complexity:
            stop = False if len(list(set(string))) < nr_symbols else True
            
    return string, complexity_0
    
    
    
def LZWStringLibrary(symbols=(1,10,5), complexity=(5,25,5), 
                     symbols_range_distribution=None, 
                     complexity_range_distribution=None,
                     iterations=1, save_csv=False, 
                     priorise_complexity=True):
    '''Serialiser generator of string libraries based on LZWStringGenerator

    Parameters
    ----------
    symbols : int or array-like, default (1,10,5)
        If integer the function will return a collection of strings of the 
        specified number of symbols. If array-like with value two or three
        it will return strings produced serially within the specified 
        range (start,stop,values). If the array is larger than three it will 
        produce strings with values within the array.
    complexity : integer or array-like, default (5,25,5)
        If integer the function will return a collection of strings with the 
        specified LZW complexity. If it is an array-like  with value two or three 
        it will return strings produced serially within the specified 
        range (start,stop,values). If the array is larger than three it will 
        produce strings with values within the array.
    symbols_range_distribution : str, default None
        Type of distribution of values within the range specified on symbols.
        Only accept the strings 'linear' or 'geometrical'.
    complexity_range_distribution : str, default None
        Type of distribution of values within the range specified on complexity.
        Only accept the strings 'linear' or 'geometrical'.
    iterations : int, default 1
        Defines the number of iterations of strings to generate within the set 
        parameters.
    save_csv: boolean, default False
        Saves the returned data frame into a CSV file.
    priorise_complexity : boolean, default True
        If true generates a string with complexity about target_complexity, 
        otherwise the generated string goes above the target complexity until 
        reach the nr_symbols.

    Returns
    -------
    CSV file and pandas.DataFrame:
        File and Data frame with columns: nr_symbols, LZW_complexity, 
        length and string; which stand for number of symbols, quantity of LZW 
        compression elements, length of the resulting string, and the string.
    '''

    # Gets the ranges of the parameters to calculate the iterations.
    if isinstance(symbols, int):
        symbols = [symbols]
    else:
        symbols = list(symbols)
    if isinstance(complexity, int):
        complexity = [complexity]
    else:
        complexity = list(complexity)
    
    len_symbols = len(symbols) >= 2 
    len_complexity = len(complexity) >= 2  
    
    if symbols_range_distribution:
        if len_symbols:
            if len(symbols) == 2:
                symbols.append(10)
            if symbols_range_distribution == 'geometrical':
                distribution_symbols = np.geomspace(*symbols)
            elif symbols_range_distribution == 'linear':
                distribution_symbols = np.linspace(*symbols)
            else:
                raise Exception('Distribution unknown.')
    else:
        distribution_symbols = symbols
                
    if complexity_range_distribution:
        if len_complexity:
            if len(complexity) == 2:
                complexity.append(10)
            if complexity_range_distribution == 'geometrical':
                distribution_complexity = np.geomspace(*complexity)
            elif complexity_range_distribution == 'linear':
                distribution_complexity = np.linspace(*complexity)
            else:
                raise Exception('Distribution unknown.')
    else:
        distribution_complexity = complexity
    
    symbols_r = np.round(distribution_symbols).astype(int)
    complexity_r = np.round(distribution_complexity).astype(int)
    
    # To avoid nested loops, an iterator product of parameters
    if iterations >= 1:
        iterations_r = range(iterations)
        iterator =  list(product(iterations_r, symbols_r, complexity_r))
        n_iter = len(iterator)
    else:
        warnings.warn('Iterations minor than 1, returns no data')
        return 

    cols = ['nr_symbols', 'LZW_complexity', 'length', 'string']
    string_lib = pd.DataFrame(columns = cols)

    for n, i in enumerate(iterator, 1):
        _, symb_i, complex_i = i
        str_, str_complex = LZWStringGenerator(symb_i, complex_i,
                                priorise_complexity=priorise_complexity)
        if not isinstance(str_, str):
            nr_symbols = 0
            str_length = 0
        else: 
            nr_symbols = len(set(str_))
            str_length = len(str_)
        df = pd.DataFrame([[nr_symbols, str_complex, str_length, str_]], 
                                        columns = cols)
        string_lib = string_lib.append(df)
        
        # To show level of progress
        print(f'\rProcessing: {n} of {n_iter}', end = '\r')
  
    # Saving of the results in a CSV file
    if save_csv:
        hd = 'StrLib_Symb'
        xt = '.csv'
        symbols = [symbols] if not isinstance(symbols, list) else symbols
        symbols_str=''
        
        for i, si in enumerate(symbols):
            s = f'{si:02}'
            sym_len = len(symbols)
            sep = '-' if (sym_len>1 and i!=sym_len-1 and i!=sym_len-2
                         ) or (sym_len==2 and i!=sym_len-1) else '_' if i==sym_len-2 else ''
            symbols_str += s + sep
        
        complexity=[complexity] if not isinstance(complexity, list) else complexity
        complexity_str=''
        for i, ci in enumerate(complexity):
            c = f'{ci}'
            cpx_len = len(complexity)
            sep = '-' if (cpx_len>1 and i!=cpx_len-1 and i!=cpx_len-2
                         ) or (cpx_len==2 and i!=cpx_len-1
                              ) else '_' if i==cpx_len-2 else ''
            complexity_str += c + sep
      
        file_name = f'{hd}({symbols_str})_LZWc({complexity_str})_Iters({iterations}){xt}'
        
        # To prevent repetited strings and strings using less symbols than 
        # required.
        string_lib = string_lib[string_lib[cols[0]]>=symbols_r[0]]
        string_lib.dropna(inplace=True)
        string_lib.drop_duplicates(subset=cols[-1], keep='first', inplace=True)
        string_lib.sort_values(cols, axis=0, ascending=True, inplace=True) 
        
        # Saving the string library to a CSV file
        string_lib.to_csv(file_name, index=False)
        print(f'\nString library saved in file: {file_name}')

    return string_lib
