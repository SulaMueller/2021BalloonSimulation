"""
@name:      ReversedProblem
@author:    Sula Spiegel
@change:    09/02/2023

@summary:   solve reversed problem (get neural activation function, if BOLD/VASO signal is known)
"""

import matplotlib.pyplot as plt
import numpy as np

from class_main_signalModel import SignalModel
from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters
from class_inputTimeline import Input_Timeline
from class_NeuralModel import Neural_Model
from class_Balloon import Balloon
from class_DependencyGenerator import DependencyGenerator

class ReversedProblem:
    def __init__(self, signal, params:Model_Parameters, signaltype='BOLD'):
        self.signal = signal
        self.params = params
        # get flow from signal
        if signaltype[0] in ['B', 'b']: flow = self.__reverseBOLD()
        elif signaltype[0] in ['V', 'v']: flow = self.__reverseVASO()
        # get activation function from balloon
        self.__reverseFlow(flow)

    ''' __reverseBOLD: get flow from known BOLD signal '''
    def __reverseBOLD(self):
        print('ToDo')

    ''' __reverseVASO: get flow from known VASO signal '''
    def __reverseVASO(self):
        print('ToDo')

    ''' __reverseFlow: get neural activation function from known flow '''
    def __reverseFlow(self, flow):
        print('ToDo')