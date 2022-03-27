# -*- coding: utf-8 -*-

"""This part is mainly for symbolic sequnce generation"""

import random
import numpy as np
from .mkc import *
from random import randint

def random_generate(start,num,power,length,capital=True,random_state=42, verbose=0):
    """
    Args:
    ------------------------------------------------------------------------------
       start(str):        Letter such as "A"
       num (int):         This is for basic symbolic generation-.generate(),which means the number of unique symbols generated.           
       power(int):        The highest number of symbols repeating.
       length(int):       The length of the symbols sequence generated.
       capital(bool):     Case or not.
       random_state(int): Ramdom seed, for basic_generate() and markovChain_generate()
    ------------------------------------------------------------------------------
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
    

def markovChain_gererate(n,current_state,no, verbose=0):
    """Apply Markov Chain to generate string sequence
    
    Args
    ------------------------------------------------------------------------------
        n(int): Number of symbols
        
        transition_prob(dict): A dict object representing the transition 
        probabilities in Markov Chain. Should be of the form: 
            {'state1': {'state1': 0.1, 'state2': 0.4}, 
             'state2': {...}}
        
        current_state(str):  The state of the current random variable.
        no: int, The number of future states to generate.
    ------------------------------------------------------------------------------
    """
    
    transition_prob = MarkovMatrix(n)
    sequence = MarkovChain(transition_prob)
    string_gen = sequence.generate_states(current_state, no)
    if verbose:
        print("generate:", string_gen)
    return string_gen
    

