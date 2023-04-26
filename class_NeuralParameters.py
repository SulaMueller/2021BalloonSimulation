"""
@name:      Neural_Parameters
@author:    Sula Spiegel
@change:    12/04/2023

@summary:   Class to store neural model parameters (read them from file first)
"""

import numpy as np
from readFile import getFileText, readValFromText
from warnUsr import warn
from class_ModelParameters import clearAttrs

class Neural_Parameters:
    def __init__(self, parameter_file):
        self.parameter_file = parameter_file
        self.__parse_parameterFile()
        clearAttrs(self, ['filetext'])
    
    ''' __parse_val: read a single value from the parameter file ''' 
    def __parse_val(self, varname, typestring='float'):
        return readValFromText(self.filetext, varname, typestring)
    
    ''' __parse_parameterFile: read all required parameters from the parameter file '''
    def __parse_parameterFile(self):
        self.filetext = getFileText(self.parameter_file)  # gets total file as string

        self.sigma = self.__parse_val('self-inhibitory connection (sigma)')
        self.mu = self.__parse_val('inhibitory-excitatory connection (mu)')
        self.lambd = self.__parse_val('inhibitory gain (lambda)')
        self.B_sigma = self.__parse_val('modulatory parameter of sigma')
        self.B_mu = self.__parse_val('modulatory parameter of mu')
        self.B_lambda = self.__parse_val('modulatory parameter of lambda')
        self.C = self.__parse_val('input weighting')
        self.c1 = self.__parse_val('c1')
        self.c2 = self.__parse_val('c2')
        self.c3 = self.__parse_val('c3')

        if any([self.B_sigma, self.B_mu, self.B_lambda]):
            raise Exception('Neural Modulation not implemented yet. Please set modulatory parameters to 0.')
 