

import matplotlib.pyplot as plt
import numpy as np
import pylops
from pylops.utils import dottest
from pylops.optimization.basic import lsqr

from class_NeuralParameters import Neural_Parameters
from class_ModelParameters import Model_Parameters
from class_inputTimeline import Input_Timeline


Cop = pylops.signalprocessing.Convolve1D(T, h=h, offset=hcenter, dtype="float32")  # operator that convolves input with wavelet
print('Convolution operator', Cop)
dottest(Cop, verb=True)  # check, that forward and adjoint operators are correct

class Ops:
    def __init__(self, \
                nparams: Neural_Parameters, \
                params: Model_Parameters, \
                input_TimeLine: Input_Timeline ):
        self.nparams = nparams
        self.params = params
        self.inputTL = input_TimeLine
        
    def neuro_excitatory(self):
        print('ToDo')
    
    def neuro_inhibitory(self):
        print('ToDo')

''' __init_matrices: initialize all matrices needed for neural model calculation '''
def __init_matrices(self):
    self.n_excitatory = np.zeros([self.params.numDepths, self.params.N])
    self.n_inhibitory = np.zeros([self.params.numDepths, self.params.N])
    self.vas = np.zeros([self.params.numDepths, self.params.N])
    self.f_arteriole = np.ones([self.params.numDepths, self.params.N])

def __getExcitatory(self, d,t):
    #yn(:,1)   = yn(:,1) + dt*(A*Xn(:,1) - MU.*Xn(:,2) + C*U.u(t,:)');
    self.n_excitatory[d,t] = self.n_excitatory[d,t-1] + self.params.dt * \
        ( self.nparams.sigma * self.n_excitatory[d,t-1] \
        - self.nparams.mu * self.n_inhibitory[d,t-1] \
        + self.nparams.C * self.inputTL.neural_input[d,t] )

def __getInhibitory(self, d,t):
    #yn(:,2)   = yn(:,2) + dt*(LAM.*(-Xn(:,2) +  Xn(:,1)));
    self.n_inhibitory[d,t] = self.n_inhibitory[d,t-1] + self.params.dt * self.nparams.lambd * ( self.n_excitatory[d,t-1] - self.n_inhibitory[d,t-1] )

def __getVasoActiveSignal(self, d,t):
    self.vas[d,t] = self.vas[d,t-1] + self.params.dt * ( self.n_excitatory[d,t-1] - self.nparams.c1 * self.vas[d,t-1])

def __getFlow(self, d,t):
    df = self.nparams.c2 * self.vas[d,t-1] - self.nparams.c3 * (self.f_arteriole[d,t-1] - 1)
    self.f_arteriole[d,t] = self.f_arteriole[d,t-1] * math.exp(self.params.dt * df / self.f_arteriole[d,t-1])

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

    
