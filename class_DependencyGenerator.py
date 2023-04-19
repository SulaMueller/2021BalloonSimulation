"""
@name:      DependencyGenerator
@author:    Sula Spiegel
@change:    20/01/2023

@summary:   get dependency of overall signal from single variables
"""

import numpy as np
import matplotlib.pyplot as plt

from class_SignalModel import Signal_Model


class DependencyGenerator:
    def __init__(self, parent:Signal_Model):
        self.parent = parent
    
    ''' calculateDependency: plot the dependency of BOLD/VASO-signal (depth-wise) on a variable <varname> 
        INPUT: 
            * varname: name of independent variable 
            * minVal, maxVal: minimum and maximum value the independent is varied between
            * EITHER: numIt OR: stepSize -> determine explicit values of the independent 
            * if the independent is a matrix: index of matrix entry (if index==[], entire matrix is varied) 
            * if the independent is in [F0, V0, tau0]: they depend on each other as {tau0 = V0/F0}; dependentVFT gives the varname that is changed as well '''
    def calculateDependency(self, varname, minVal, maxVal, numIt=100, stepSize=-1, index=[], dependentVFT='tau0'):
        # prepare
        self.__include(varname, minVal, maxVal, numIt, stepSize, index, dependentVFT)
        self.__initMatrice()
        # save original value of <varname>
        _, bold = self.parent.params.findVarname(varname)  # find out where the variable is stored
        orig_val = self.parent.params.getVarValue(varname, bold)
        # execute
        self.__iterate()
        self.__normalize()
        self.__plotMaxima()
        # set <varname> back to original value
        self.__changeSignal(orig_val)

    ''' __include: write all input variables into self '''
    def __include(self, varname, minVal, maxVal, numIt, stepSize, index, dependentVFT):
        self.varname = varname
        self.minVal = minVal
        self.maxVal = maxVal
        self.numIt = numIt
        if stepSize==-1: self.stepsize = (self.maxVal - self.minVal) / (self.numIt - 1 )
        else:
            self.stepsize = stepSize
            self.numIt = (self.maxVal - self.minVal) / self.stepsize + 1
        self.index = index
        self.dependentVFT = dependentVFT

    ''' __initMatrice: initialize all result matrice and the x-axis '''
    def __initMatrice(self):
        # initialize matrice 
        for attr in ['BOLD_max', 'VASO_max', 'BOLD_timestamps', 'VASO_timestamps']:
            if not hasattr(self, attr): setattr(self, attr, np.zeros([self.parent.params.numDepths, self.numIt]))  
        # get x-axis (all values over which will be iterated)
        self.x = np.linspace(self.minVal, self.maxVal, self.numIt)

    ''' __changeSignal: update the signal model with a new value for <varname> '''
    def __changeSignal(self, newVal):
        self.parent.params.changeVar(self.varname, newVal, self.index, self.dependentVFT)
        self.parent.createModelInstances()

    ''' __iterate: function to iterate over all variations of the independent variable (and save resulting signal changes)
            1) change variable in Model_Parameters
            2) adapt signal model
            3) save maximum of BOLD/VASO-signals and time of their occurence
            4) iterate 1-3 '''
    def __iterate(self):
        for i in range(0, self.numIt):  # iterate through x-axis
            self.__changeSignal(self.x[i])  # change model
            # save maximum signal values and time of their occurence
            for d in range(0, self.parent.params.numDepths):
                self.BOLD_max[d,i] = np.max(self.parent.balloon.bold.BOLDsignal[d,:])
                self.VASO_max[d,i] = np.max(self.parent.balloon.bold.VASOsignal[d,:])
                self.BOLD_timestamps[d,i] = np.argmax(self.parent.balloon.bold.BOLDsignal[d,:])
                self.VASO_timestamps[d,i] = np.argmax(self.parent.balloon.bold.VASOsignal[d,:])
    
    ''' __normalize: normalize BOLD_max, VASO_max to [0,1] (save former max to bmax,vmax) '''
    def __normalize(self):
        self.bmax = np.zeros([self.parent.params.numDepths])  # maxima of BOLD-signal, depthwise
        self.vmax = np.zeros([self.parent.params.numDepths])  # maxima of VASO-signal, depthwise
        for d in range(0, self.parent.params.numDepths):
            self.bmax[d] = np.max(self.BOLD_max[d,:])
            self.vmax[d] = np.max(self.VASO_max[d,:])
            self.BOLD_max[d,:] = self.BOLD_max[d,:]/self.bmax[d]
            self.VASO_max[d,:] = self.VASO_max[d,:]/self.vmax[d]

    ''' __plotMaxima: function to plot result from __iterate (depth-wise) '''
    def __plotMaxima(self):
        _, axs = plt.subplots(self.parent.params.numDepths, 1)  
        for d in range(0, self.parent.params.numDepths):
            ax = axs[d]
            ax.plot(self.x, self.BOLD_max[d,:], label=f'BOLD (max={self.bmax[d]})')
            ax.plot(self.x, self.VASO_max[d,:], label=f'VASO (max={self.vmax[d]})')
            ax.grid(True)
            ax.set_ylabel('signal(d='+str(d+1)+')')
            ax.legend()
        ax.set_xlabel(self.varname)
        
    

    

    



