"""
@name:      Neural_Model
@author:    Sula Spiegel
@change:    12/08/2021

@summary:   Class to give a neural input
@reference: freely adapted from Havlicek2020
"""

import numpy as np
import math
from class_NeuralParameters import Neural_Parameters
from class_ModelParameters import Model_Parameters

class Neural_Model:
    def __init__(self, nparams: Neural_Parameters, params: Model_Parameters):
        print('stuff')