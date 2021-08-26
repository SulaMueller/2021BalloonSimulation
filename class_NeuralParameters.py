"""
@name:      Neural_Parameters
@author:    Sula Spiegel
@change:    12/08/2021

@summary:   Class to store neural model parameters (read them from file first)
"""

import numpy as np
from readFile import getFileText, readFloatFromText, readMatrixFromText
from warnings import warn

class Neural_Parameters:
    def __init__(self, parameter_file):
        self.parameter_file = parameter_file
        self.__parse_parameterFile()
    
    ''' __parse_val: read a single value from the parameter file ''' 
    def __parse_val(self, varname):
        return readFloatFromText(self.filetext, varname)
    
    ''' __parse_parameterFile: read all required parameters from the parameter file '''
    def __parse_parameterFile(self):
        self.filetext = getFileText(self.parameter_file)  # gets total file as string

        self.sigma = self.__parse_val('self-inhibitory connection (sigma)')[0]
        self.mu = self.__parse_val('inhibitory-excitatory connection (mu)')[0]
        self.lambd = self.__parse_val('inhibitory gain (lambda)')[0]
        self.B_sigma = self.__parse_val('modulatory parameter of sigma')[0]
        self.B_mu = self.__parse_val('modulatory parameter of mu')[0]
        self.B_lambda = self.__parse_val('modulatory parameter of lambda')[0]
        self.C = self.__parse_val('input weighting')[0]
        self.c1 = self.__parse_val('c1')[0]
        self.c2 = self.__parse_val('c2')[0]
        self.c3 = self.__parse_val('c3')[0]

        if any([self.B_sigma, self.B_mu, self.B_lambda]):
            raise Exception('Neural Modulation not implemented yet. Please set modulatory parameters to 0.')
 