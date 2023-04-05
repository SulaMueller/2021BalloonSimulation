"""
@name:      InverseProblem
@author:    Sula Spiegel
@change:    09/02/2023

@summary:   solve inverse problem (get neural activation function, if BOLD/VASO signal is known)
"""

import matplotlib.pyplot as plt
import numpy as np

from class_SignalModel import Signal_Model
from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters
from class_inputTimeline import Input_Timeline
from class_NeuralModel import Neural_Model
from class_Balloon import Balloon

class InverseProblem:
    def __init__(self, signal:Signal_Model, params:Model_Parameters, signaltype='BOLD'):
        self.signal = signal
        self.params = params
        # get flow from signal
        if signaltype[0] in ['B', 'b']: self.flow = self.__inverseBOLD()
        elif signaltype[0] in ['V', 'v']: self.flow = self.__inverseVASO()
        # get neural activation function from flow
        self.neural = self.__inverseFlow()
    
    ''' =============================  SET-FUNCTIONS  ======================== '''
    
    ''' setFlow: set function for flow (public function) '''
    def setFlow(self, flow):
        self.flow = flow
    
    ''' setNeural: set function for neural activation (public function) '''
    def setNeural(self, neural):
        self.neural = neural
    
    ''' =============================  GET FLOW  ======================== '''

    ''' __inverseBOLD: get flow from known BOLD signal '''
    def __inverseBOLD(self):
        print('ToDo')
        flow = 0
        self.setFlow(flow)

    ''' __inverseVASO: get flow from known VASO signal '''
    def __inverseVASO(self):
        print('ToDo')
        flow = 0
        self.setFlow(flow)

    ''' =============================  GET NEURAL  ======================== '''

    ''' __inverseFlow: get neural activation function from known flow '''
    def __inverseFlow(self):
        print('ToDo')
        neural = self.flow
        self.setNeural(neural)