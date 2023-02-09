"""
@name:      DependencyGenerator
@author:    Sula Spiegel
@change:    20/01/2023

@summary:   get dependency of overall signal from single variables
"""

import numpy as np
import matplotlib.pyplot as plt

from class_main_signalModel import SignalModel


class DependencyGenerator:
    def __init__(self, parent:SignalModel):
        self.parent = parent

    ''' __iterate: function to iterate over all variations of the independent variable (and save resulting signal changes)
            1) change variable in Model_Parameters
            2) adapt signal model
            3) save maximum of BOLD/VASO-signals and time of their occurence
            4) iterate 1-3 '''
    def __iterate(self, index, dependentVFT):
        # iterate through x-axis
        for i in range(0, self.numIt):
            # change model
            self.parent.params.changeVar(self.varname, self.x[i], index, dependentVFT)
            self.parent.createModelInstances()
            # save maximum signal values and time of their occurence
            for d in range(0, self.parent.params.numDepths):
                self.BOLD_max[d,i] = np.max(self.parent.balloon.bold.BOLDsignal[d,:])
                self.VASO_max[d,i] = np.max(self.parent.balloon.bold.VASOsignal[d,:])
                self.BOLD_timestamps[d,i] = np.argmax(self.parent.balloon.bold.BOLDsignal[d,:])
                self.VASO_timestamps[d,i] = np.argmax(self.parent.balloon.bold.VASOsignal[d,:])
    
    ''' __normalize: normalize BOLD_max, VASO_max to [0,1] (save former max to bmax,vmax) '''
    def __normalize(self):
        self.bmax = np.max(self.BOLD_max)
        self.vmax = np.max(self.VASO_max)
        self.BOLD_max = self.BOLD_max/self.bmax
        self.VASO_max = self.VASO_max/self.vmax

    ''' __plotMaxima: function to plot result from __iterate (depth-wise) '''
    def __plotMaxima(self):
        _, axs = plt.subplots(self.parent.params.numDepths, 1)  
        for d in range(0, self.parent.params.numDepths):
            ax = axs[d]
            ax.plot(self.x, self.BOLD_max[d,:], label=f'BOLD (max={self.bmax})')
            ax.plot(self.x, self.VASO_max[d,:], label=f'VASO (max={self.vmax})')
            ax.grid(True)
            ax.set_ylabel('signal(d='+str(d+1)+')')
        ax.set_xlabel(self.varname)
        ax.legend()
    
    ''' plotDependency: plot the dependency of BOLD/VASO-signal (depth-wise) on a variable <varname> 
        INPUT: 
            * varname: name of independent variable 
            * minVal, maxVal: minimum and maximum value the independent is varied between
            * EITHER: numIt OR: stepSize -> determine explicit values of the independent 
            * if the independent is a matrix: index of matrix entry (if index==[], entire matrix is varied) 
            * if the independent is in [F0, V0, tau0]: they depend on each other as {tau0 = V0/F0}; dependentVFT gives the varname that is changed as well '''
    def plotDependency(self, varname, minVal, maxVal, numIt=100, stepSize=-1, index=[], dependentVFT='tau0'):
        # include variables
        self.varname = varname
        self.minVal = minVal
        self.maxVal = maxVal
        self.numIt = numIt
        if stepSize==-1: self.stepsize = (self.maxVal - self.minVal) / (self.numIt - 1 )
        else:
            self.stepsize = stepSize
            self.numIt = (self.maxVal - self.minVal) / self.stepsize + 1
        # get x-axis
        self.x = np.linspace(self.minVal, self.maxVal, self.numIt)
        # initialize matrice 
        self.BOLD_max = np.zeros([self.parent.params.numDepths, self.numIt])
        self.VASO_max = np.zeros([self.parent.params.numDepths, self.numIt])
        self.BOLD_timestamps = np.zeros([self.parent.params.numDepths, self.numIt])
        self.VASO_timestamps = np.zeros([self.parent.params.numDepths, self.numIt])
        # execute
        self.__iterate(index, dependentVFT)
        self.__normalize()
        self.__plotMaxima()

