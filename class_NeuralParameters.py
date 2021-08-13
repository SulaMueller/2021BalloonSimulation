"""
@name:      Neural_Parameters
@author:    Sula Spiegel
@change:    12/08/2021

@summary:   Class to store neural model parameters (read them from file first)
"""

import numpy as np
from readFile import getFileText, readFloatFromText, readMatrixFromText

class Neural_Parameters:
    def __init__(self, parameter_file):
        self.__parse_parameterFile(parameter_file)
    
    ''' __parse_val: read a single value from the parameter file ''' 
    def __parse_val(self, varname):
        return readFloatFromText(self.filetext, varname)
    
    ''' __parse_parameterFile: read all required parameters from the parameter file '''
    def __parse_parameterFile(self, filename):
        self.filetext = getFileText(filename)  # gets total file as string

        self.sigma = self.__parse_val('self-inhibitory connection (sigma)')
        self.mu = self.__parse_val('inhibitory-excitatory connection (mu)')
        self.lambd = self.__parse_val('inhibitory gain (lambda)')
        self.B_sigma = self.__parse_val('modulatory parameter of sigma')
        self.B_mu = self.__parse_val('modulatory parameter of mu')
        self.B_lambda = self.__parse_val('modulatory parameter of lambda')