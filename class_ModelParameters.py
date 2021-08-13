
"""
@name:      Model_Parameters
@author:    Sula Spiegel
@change:    05/08/2021

@summary:   Class to store general model parameters (read them from file first)
"""

import numpy as np
from warnings import warn
from readFile import getFileText, readFloatFromText, readMatrixFromText

class Model_Parameters:
    def __init__(self, parameter_file, f_arteriole=None):
        self.COMPARTMENTS = {"ARTERIOLE", "VENULE", "VEIN"}
        self.ARTERIOLE = 0
        self.VENULE = 1
        self.VEIN = 2
        self.INFLATION = 0  # for visco-elastic time constant
        self.DEFLATION = 1

        self.numCompartments = len(self.COMPARTMENTS)

        self.__parse_parameterFile(parameter_file)
        self.__check_flowSanity()

        if f_arteriole is None: self.__read_fArteriole()
        else: self.set_fArteriole(f_arteriole)

    ''' __init_matrices: initialize all matrices needed '''
    def __init_matrices(self):
        self.V0 = np.zeros([self.numCompartments, self.numDepths])
        self.F0 = np.zeros([self.numCompartments, self.numDepths])
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
            'TE': self.__parse_val("TE"),
            'gamma0': 2*np.pi*42.58*pow(10,6)
        }
    
    ''' __compareValues: compare two values with each other and make sure they are the same 
                            * set both to k_default if they are not the same '''
    def __compareValues(self, valuematrix, k1, k2, k_default, d, errormessage=''):
        if valuematrix[k1, d] != valuematrix[k2, d]:
            if valuematrix[k1, d] == 0: 
                valuematrix[k1, d] = valuematrix[k2, d]
            elif valuematrix[k2, d] == 0:
                valuematrix[k2, d] = valuematrix[k1, d]
            else:
                if len(errormessage) > 0: warn(errormessage)
                valuematrix[k1, d] = valuematrix[k_default, d]
                valuematrix[k2, d] = valuematrix[k_default, d]
    
    ''' __check_flowSanity: make sure, that F0_v == F0_a and F0_d(K) = F0_v(K) (K: deepest layer) '''
    def __check_flowSanity(self):
        # F0_venule = F0_arteriole
        k_default = self.VENULE
        for d in range(0, self.numDepths):
            errormessage = f"F0_arteriole({d}) != F0_venule({d}). Setting both to F0_venule (= {self.F0[k_default, d]})"
            self.__compareValues(self.F0, self.ARTERIOLE, self.VENULE, k_default, d, errormessage)
            
        # F0_vein(numDepths-1) = F0_venule(numDepths-1)
        k_default = self.VENULE
        d = self.numDepths -1
        errormessage = f"F0_vein(K) != F0_venule(K). Setting both to F0_venule (= {self.F0[k_default, d]})"
        self.__compareValues(self.F0, self.VENULE, self.VEIN, k_default, d, errormessage)

    ''' set_fArteriole: assign a given array <f_new> to f_arteriole '''
    def set_fArteriole(self, f_new):
        self.f_arteriole = f_new

    ''' __read_fArteriole: initial assignement of f_arteriole (read from parameter file)'''
    def __read_fArteriole(self):
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
                if t0 < self.N:
                    f_arteriole[0, d, t0:t1] = flowparams[flowaxis, d, s]
        self.set_fArteriole(f_arteriole)
    