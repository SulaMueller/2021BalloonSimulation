
"""
@name:      Model_Parameters
@author:    Sula Spiegel
@change:    05/08/2021

@summary:   Class to store general model parameters (read them from file first)
"""

import numpy as np
from readFile import getFileText, readFloatFromText, readMatrixFromText

class Model_Parameters:
    def __init__(self, parameter_file):
        self.numCompartments = 3   # arteriole, venule, vein
        self.COMPARTMENTS = {"ARTERIOLE", "VENULE", "VEIN"}
        self.ARTERIOLE = 0
        self.VENULE = 1
        self.VEIN = 2
        self.INFLATION = 0  # for visco-elastic time constant
        self.DEFLATION = 1
        self.gamma0 = 2*np.pi*42.58*pow(10,6)
        self.__parse_parameterFile(parameter_file)
        self.__init_fArteriole()

    ''' __init_matrices: initialize all matrices needed '''
    def __init_matrices(self):
        self.V0 = np.empty([self.numCompartments, self.numDepths])
        self.F0 = np.empty([self.numCompartments, self.numDepths])
        self.alpha = np.empty([self.numCompartments, self.numDepths])
        self.vet = np.empty([self.numCompartments, self.numDepths, 2])  # visco-elastic time-constants for in-/deflation
    
    ''' __parse_val: read a single value from the parameter file ''' 
    def __parse_val(self, varname):
        return readFloatFromText(self.filetext, varname)
    
    ''' __parse_matrix: read an entire matrix from the parameter file '''
    def __parse_matrix(self, varname, nVar, outmatrix):
        return readMatrixFromText(self.filetext, varname, nVar, self.numCompartments, self.numDepths, outmatrix)

    ''' __parse_parameterFile: read all required parameters from the parameter file '''
    def __parse_parameterFile(self, filename):
        self.filetext = getFileText(filename)  # gets total file as string
        self.N = int(self.__parse_val("number of time points"))
        self.dt = self.__parse_val("time integration step (dt)")
        self.numDepths = int(self.__parse_val("number of depth levels"))
        self.__init_matrices()
        self.__parse_matrix('V0', 0, self.V0)
        self.__parse_matrix('F0', 1, self.F0)
        self.__parse_matrix('alpha', 0, self.alpha)
        self.__parse_matrix('visco-elastic time constants', -1, self.vet)
        numBOLDparams = int(self.__parse_val("number of BOLD parameters"))
        boldparams_tmp = readMatrixFromText(self.filetext, 'BOLD', 0, self.numCompartments, numBOLDparams)
        self.E0 = boldparams_tmp[:,0]
        self.B0 = self.__parse_val("B0")
        self.n = self.__parse_val("n-ratio")
        self.boldparams = {
            'epsilon': boldparams_tmp[:,1],
            'Hct': boldparams_tmp[:,2],
            'r0': boldparams_tmp[:,3],
            'dXi': self.__parse_val("dXi"),
            'TE': self.__parse_val("TE")
        }

    ''' set_fArteriole: assign a given array <f_new> to f_arteriole '''
    def set_fArteriole(self, f_new):
        self.f_arteriole = f_new

    ''' __init_fArteriole: initial assignement of f_arteriole (read from parameter file)'''
    def __init_fArteriole(self):
        numSections = int(self.__parse_val("number of flow sections"))
        numAxis = 2  # time, flow
        timeaxis = 0
        flowaxis = 1
        flowparams = np.empty([numAxis, self.numDepths, numSections])
        readMatrixFromText(self.filetext, 'flow', -1, numAxis, self.numDepths, flowparams)

        f_arteriole = np.empty([1, self.numDepths, self.N])  # need first dim for easy copying to flow-array
        for d in range(0, self.numDepths):
            for s in range(0, numSections):
                t0 = int(flowparams[timeaxis, d, s])
                if s < numSections - 1: t1 = int(flowparams[timeaxis, d, s+1])
                else: t1 = self.N
                f_arteriole[0, d, t0:t1] = flowparams[flowaxis, d, s]
        self.set_fArteriole(f_arteriole)
    