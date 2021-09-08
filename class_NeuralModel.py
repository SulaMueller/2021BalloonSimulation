"""
@name:      Neural_Model
@author:    Sula Spiegel
@change:    12/08/2021

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
    
    ''' __check_input: check, whether given input structure contains a neural activation function '''
    def __check_input(self):
        if not self.inputTL.available_input[self.inputTL.INDEX_NEURO]:
            warn('No neural input function given. Neuronal model not calculated!')
            return False
        return True

    ''' __init_matrices: initialize all matrices needed for neural model calculation '''
    def __init_matrices(self):
        self.n_excitatory = np.zeros([self.params.numDepths, self.params.N])
        self.n_inhibitory = np.zeros([self.params.numDepths, self.params.N])
        self.vas = np.zeros([self.params.numDepths, self.params.N])
        self.f_arteriole = np.ones([self.params.numDepths, self.params.N])

    def __getExcitatory(self, d,t):
        #yn(:,1)   = yn(:,1) + dt*(A*Xn(:,1) - MU.*Xn(:,2) + C*U.u(t,:)');
        self.n_excitatory[d,t] = \
              self.n_excitatory[d,t-1] \
            + self.params.dt * ( \
                  self.nparams.sigma * self.n_excitatory[d,t-1] \
                - self.nparams.mu * self.n_inhibitory[d,t-1] \
                + self.nparams.C * self.inputTL.neural_input[d,t] \
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
        for t in range(1, self.params.N):
            for d in range(0, self.params.numDepths):
                self.__getExcitatory(d,t)
                self.__getInhibitory(d,t)
                self.__getVasoActiveSignal(d,t)
                self.__getFlow(d,t)
        self.inputTL.set_input(self.f_arteriole, self.inputTL.INDEX_FLOW)
    
    ''' __longName: get human readable name of specific function '''
    def __longName(self, funcname):
        longnames = {
            'n_excitatory': 'Neuronal Activation Function',
            'f_arteriole': 'Flow Response'
        }
        return longnames[funcname]

    ''' __getFunc: return a function (n_excitatory, f_arteriole) on specified depth
                    (return all depths if depth==-1) '''
    def __getFunc(self, funcname, depth):
        if not hasattr(self, funcname):
            warn(f'{self.__longName(funcname)} cannot be returned because it has not been calculated!')
            return None
        if depth==-1:  
            return getattr(self, funcname)
        return getattr(self, funcname)[depth,:]

    def get_activationFunction(self, depth=-1):
        return self.__getFunc('n_excitatory', depth)
    def get_flowResponse(self, depth=-1):
        return self.__getFunc('f_arteriole', depth)
    
    ''' __plotFunc: plot a function (n_excitatory, f_arteriole) on specified depth
                    (plot all depths if depth==-1) '''
    def __plotFunc(self, funcname, depth):
        if not hasattr(self, funcname):
            warn(f'{self.__longName(funcname)} cannot be plotted because it has not been calculated!')
            return
        time = np.linspace(0, self.params.N, self.params.N)
        numLines = self.params.numDepths
        if depth > -1: numLines = 1
        numColumns = 1
        _, axs = plt.subplots(numLines, numColumns) 
        for d in range(0, numLines):
            if hasattr(axs, '__len__'): a = axs[d]
            else: a = axs
            if d==0: 
                a.set_title(self.__longName(funcname))
            a.grid(True)
            if d==numLines-1:
                a.set_xlabel('t')
            if depth > -1:
                d = depth
            a.plot(time, getattr(self, funcname)[d,:])
    
    def plot_activationFunction(self, depth=-1):
        self.__plotFunc('n_excitatory', depth)
    def plot_flowResponse(self, depth=-1):
        self.__plotFunc('f_arteriole', depth) 
        
