
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

        self.priorityOfCompartments = [self.VEIN, self.ARTERIOLE, self.VENULE]
        self.numCompartments = len(self.COMPARTMENTS)
        self.__parse_parameterFile(parameter_file)
        self.__check_flowSanity()

        if f_arteriole is None: self.__read_fArteriole()
        else: self.set_fArteriole(f_arteriole)

        delattr(self, 'priorityOfCompartments')
        

    ''' __init_matrices: initialize all matrices needed '''
    def __init_matrices(self):
        self.V0 = np.zeros([self.numCompartments, self.numDepths])
        self.F0 = np.zeros([self.numCompartments, self.numDepths])
        self.tau0 = np.zeros([self.numCompartments, self.numDepths])
        self.alpha = np.ones([self.numCompartments, self.numDepths])
        self.vet = np.ones([self.numCompartments, self.numDepths, 2])  # visco-elastic time-constants for in-/deflation
    
    ''' __parse_val: read a single value from the parameter file ''' 
    def __parse_val(self, varname):
        return readFloatFromText(self.filetext, varname)
    
    ''' __parse_matrix: read an entire matrix from the parameter file '''
    def __parse_matrix(self, varname, nVar, outmatrix):
        return readMatrixFromText(self.filetext, varname, nVar, self.numCompartments, self.numDepths, outmatrix)

    ''' __parse_parameterFile: read all required parameters from the parameter file '''
    def __parse_parameterFile(self, filename):
        # get total file as string
        self.filetext = getFileText(filename)  
        # get basic variables
        self.N = int(self.__parse_val("number of time points"))
        self.dt = self.__parse_val("time integration step (dt)")
        self.numDepths, haveNumDepths = self.__parse_val("number of depth levels")
        if not haveNumDepths: 
            warn('Number of Depth levels not defined.')
            self.numDepths = 6
        self.numDepths = int(self.numDepths)
        # get balloon matrices
        self.__init_matrices()
        self.__parse_matrix('V0', 0, self.V0)
        self.__parse_matrix('F0', 0, self.F0)
        self.__parse_matrix('t0', 0, self.tau0)
        self.__parse_matrix('alpha', 0, self.alpha)
        self.__parse_matrix('visco-elastic time constants', -1, self.vet)
        # get BOLD-parameters
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
    
    ''' __listIncludesCompartment: return, whether a specific compartment is included in a list
                * example for k_list: [k1,k2] '''
    def __listIncludesCompartment(self, k_list, compartment):
        return any(k == compartment for k in k_list)
    
    ''' __needDeeperLayerFormula: F0 resting state condition requires deeper layer for veins (except deepest layer) '''
    def __needDeeperLayerFormula(self, k_list, d):
        return self.__listIncludesCompartment(k_list, self.VEIN) and d<self.numDepths-1
    
    ''' __getOtherCompartment: extract k that is NOT the given compartment from a given list '''
    def __getOtherCompartment(self, k_list, compartment):
        for k in k_list: 
            if k != compartment: return k
    
    ''' __flowMeetsCondition: check if resting state conditions are met for two specific F0s'''
    def __flowMeetsCondition(self, k1, k2, d):
        if not self.__needDeeperLayerFormula([k1, k2], d):
            return self.F0[k1,d] == self.F0[k2,d]
        else:
            k_other = self.__getOtherCompartment([k1, k2], self.VEIN)
            return self.F0[self.VEIN,d] == self.F0[k_other,d] + self.F0[self.VEIN,d+1]
    
    ''' __findBlueprintVal: return the compartment that will be used as blueprint 
            -> compartment that has been changed since init of matrix or default, if both '''
    def __findBlueprintVal(self, matrix, k1, k2, k_default, d, initVal):
        # both or none has been filled -> return k_default
        if matrix[k1,d] == initVal and matrix[k2,d] == initVal or \
            matrix[k1,d] != initVal and matrix[k2,d] != initVal: 
            return k_default
        # one has been filled and the other one not -> return the changed index
        for k in [k1, k2]: 
            if matrix[k,d] != initVal: return k
    
    ''' __makeFlowMeetCondition_oneCompartment: change value of one compartment so that flow meets resting state conditions '''
    def __makeFlowMeetCondition_oneCompartment(self, k1, k2, k_default, d, initVal):
        # check for specific value, that k_default exists
        k_blueprint = self.__findBlueprintVal(self.F0, k1, k2, k_default, d, initVal)
        # if not in higher layers of vein, set both equal
        if not self.__needDeeperLayerFormula([k1, k2], d):
            self.F0[k1,d] = self.F0[k_blueprint,d]
            self.F0[k2,d] = self.F0[k_blueprint,d]
        else:  
            # for higher layers of vein, need different formula
            k_other = self.__getOtherCompartment([k1, k2], self.VEIN)
            if k_blueprint == k_other:  
                # change value of vein
                self.F0[self.VEIN,d] = self.F0[k_other,d] + self.F0[self.VEIN,d+1]
            else:
                # change value of other
                self.F0[k_other,d] = self.F0[self.VEIN,d] - self.F0[self.VEIN,d+1]
    
    ''' __makeFlowMeetConditions: change value of all compartments so that flow meets resting state conditions '''
    def __makeFlowMeetConditions(self, k_blueprint, initVal):
        for d in range(self.numDepths-1, -1, -1):
            for k in range(0, self.numCompartments):
                if k==k_blueprint: continue
                self.__makeFlowMeetCondition_oneCompartment(k, k_blueprint, k_blueprint, d, initVal)

    ''' __haveCompartment: return True, if entire compartment of matrix is not in initial matrix conditions '''
    def __haveCompartment(self, matrix, compartment, initVal):
        return sum(matrix[compartment,:]) != self.numDepths * initVal

    ''' __getInitialFlow: make sure, all F0s meet steady-state conditions '''
    def __getInitialFlow(self):
        # check, which compartments are defined
        initVal = 0  # since matrices initialised as 0
        haveCompartments = np.zeros([self.numCompartments], 'bool')
        for k in range(0, self.numCompartments):
            haveCompartments[k] = self.__haveCompartment(self.F0, k, 1)
        # define the ground truth compartment
        if sum(haveCompartments) == 0: 
            # if no compartment has values, set F0_arteriole to ones and use as basis
            k_blueprint = self.ARTERIOLE
            self.F0[self.ARTERIOLE,:] = 1
        else:
            # define priority of compartments, then use the available one with highest priority as basis
            for k in range(0, self.numCompartments):
                if haveCompartments[self.priorityOfCompartments[k]]:
                    k_blueprint = k
                    break
        # make sure all conditions are met
        self.__makeFlowMeetConditions(k_blueprint, initVal)

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
    