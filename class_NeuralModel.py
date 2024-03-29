"""
@name:      Neural_Model
@author:    Sula Spiegel
@change:    12/04/2023

@summary:   Class to give a neural response function of a stimulus
@input:     * Neural_Parameters = class to summarize parameters of response function
            * Model_Parameters = parameters for ballon, needed for general params (nTimepoints, nLayers etc)
            * neural stimulus function
@output:    arterial flow in response to neural stimulus
@reference: freely adapted from Havlicek2020
"""

import numpy as np
import math
from warnUsr import warn
import matplotlib.pyplot as plt
from class_NeuralParameters import Neural_Parameters
from class_ModelParameters import Model_Parameters
from class_inputTimeline import Input_Timeline

class Neural_Model:
    def __init__(self, \
                nparams: Neural_Parameters, \
                params: Model_Parameters, \
                input_TimeLine: Input_Timeline ):
        self.nparams = nparams
        self.params = params
        self.inputTL = input_TimeLine
        
        if not self.__check_input(): return
        self.__get_neuralModel()
    
    ''' __check_input: check, whether given input structure contains a stimulus function '''
    def __check_input(self):
        if not self.inputTL.available_input[self.inputTL.INDEX_STIMULUS]:
            warn('No stimulus given. Neuronal model not calculated!')
            return False
        return True

    ''' __init_matrices: initialize all matrices needed for neural model calculation '''
    def __init_matrices(self):
        self.n_excitatory = np.zeros([self.params.numDepths, self.params.nT])
        self.n_inhibitory = np.zeros([self.params.numDepths, self.params.nT])
        self.vas = np.zeros([self.params.numDepths, self.params.nT])
        self.f_arteriole = np.ones([self.params.numDepths, self.params.nT])

    def __getExcitatory(self, d,t):
        #yn(:,1)   = yn(:,1) + dt*(A*Xn(:,1) - MU.*Xn(:,2) + C*U.u(t,:)');
        self.n_excitatory[d,t] = \
              self.n_excitatory[d,t-1] \
            + self.params.dt * ( \
                  self.nparams.sigma * self.n_excitatory[d,t-1] \
                - self.nparams.mu * self.n_inhibitory[d,t-1] \
                + self.nparams.C * self.inputTL.stimulus[d,t] \
              )
    
    def __getInhibitory(self, d,t):
        #yn(:,2)   = yn(:,2) + dt*(LAM.*(-Xn(:,2) +  Xn(:,1)));
        self.n_inhibitory[d,t] = \
              self.n_inhibitory[d,t-1] \
            + self.params.dt * self.nparams.lambd * \
            ( self.n_excitatory[d,t-1] - self.n_inhibitory[d,t-1] )
    
    def __getVasoActiveSignal(self, d,t):
        self.vas[d,t] = \
              self.vas[d,t-1] \
            + self.params.dt * ( \
                  self.n_excitatory[d,t-1] \
                - self.nparams.c1 * self.vas[d,t-1] \
              )
    
    def __getFlow(self, d,t):
        df = self.nparams.c2 * self.vas[d,t-1] - self.nparams.c3 * (self.f_arteriole[d,t-1] - 1)
        self.f_arteriole[d,t] = \
              self.f_arteriole[d,t-1] \
            * math.exp(self.params.dt * df / self.f_arteriole[d,t-1])

    ''' __get_neuralModel: calculate neural model from given model parameters and neural input function '''    
    def __get_neuralModel(self):
        self.__init_matrices()
        for t in range(1, self.params.nT):
            for d in range(0, self.params.numDepths):
                self.__getExcitatory(d,t)
                self.__getInhibitory(d,t)
                self.__getVasoActiveSignal(d,t)
                self.__getFlow(d,t)
        self.inputTL.set_input(self.f_arteriole, self.inputTL.INDEX_FLOW)

        
