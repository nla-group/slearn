# -*- coding: utf-8 -*-

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

"""This part is mainly for symbolic sequence generation"""

import random
import numpy as np
from .mkc import *
from random import randint

def random_generate(start, num, power,length,capital=True,random_state=42, verbose=0):
    """
    Parameters
    ----------
    start : str: 
       Letter such as "A"

    num : int: 
        This is for basic symbolic generation-.generate(),which means the number of unique symbols generated.           

    power : int:   
        The highest number of symbols repeating.

    length : int:   
        The length of the symbols sequence generated.

    capital : bool:  
        Case or not.

    random_state : int: 
        Ramdom seed, for basic_generate() and markovChain_generate()

    """
    
    random.seed(random_state)
    np.random.seed(random_state)
        
    if num > 256: 
        raise symbolicError('please make sure the num < 256')

        
    slow = 97; shigh = 97 + 25
    
    if capital:
        slow,shigh = 97 - 32, 97 + 25 -32
        
    numerical_sequence = [ord(start)] + random.sample(range(slow,shigh), num)
    string_unique = [chr(i) for i in numerical_sequence]

    ord_random = np.random.randint(low = 1, high = num)
    dict_ord = dict(zip(string_unique, ord_random*[power]+(num-ord_random+1)*[1]))
    list_gen = list()
    for i in string_unique:
        for j in range(dict_ord[i]):
            list_gen.append(i)
    string_gen = []
    for i in range(round(length / len(list_gen))):
        string_gen += list_gen
    if verbose:
        print("generate:", "".join(string_gen))
    return "".join(string_gen)
    

    
def mkc_gererate(n, current_state, no, verbose=0):
    """Apply Markov Chain to generate string sequence
    
    Parameters
    ----------
    n : int: 
        Number of symbols

    transition_prob : dict: 
        A dict object representing the transition 
        
        probabilities in Markov Chain. Should be of the form: 
            {'state1': {'state1': 0.1, 'state2': 0.4}, 
             'state2': {...}}

    current_state : str:  
        The state of the current random variable.
    
    no: int,
        The number of future states to generate.
    """
    
    transition_prob = markovMatrix(n)
    sequence = markovChain(transition_prob)
    string_gen = sequence.generate_states(current_state, no)
    if verbose:
        print("generate:", string_gen)
    return string_gen
    

