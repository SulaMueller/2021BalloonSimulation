
"""
@name:      Model_Parameters
@author:    Sula Spiegel
@change:    05/08/2021

@summary:   Class to store general model parameters (read them from file first)
"""

import numpy as np
from warnings import warn
from readFile import getFileText, readFloatFromText, readStringFromText, readMatrixFromText

class Model_Parameters:
    def __init__(self, parameter_file: str):
        self.COMPARTMENTS = {"ARTERIOLE", "VENULE", "VEIN"}
        self.ARTERIOLE = 0
        self.VENULE = 1
        self.VEIN = 2
        self.INFLATION = 0  # for visco-elastic time constant
        self.DEFLATION = 1

        self.numCompartments = len(self.COMPARTMENTS)
        self.__parse_parameterFile(parameter_file)
        self.__completeInput()
        
# ---------------------------------  EXPLICIT READ-IN  ----------------------------------------
    ''' __init_matrices: initialize all matrices needed '''
    def __init_matrices(self):
        self.V0 = np.zeros([self.numCompartments, self.numDepths])
        self.F0 = np.zeros([self.numCompartments, self.numDepths])
        self.tau0 = np.zeros([self.numCompartments, self.numDepths])
        self.alpha = np.ones([self.numCompartments, self.numDepths])
        self.vet = np.ones([self.numCompartments, self.numDepths, 2])  # visco-elastic time-constants for in-/deflation
        self.__initVals = {
            'F0' : 0,
            'V0' : 0,
            'tau0' : 0,
            'alpha' : 1,
            'vet' : 1 }

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
        self.N = int(self.__parse_val("number of time points")[0])
        self.dt = self.__parse_val("time integration step (dt)")[0]
        self.numDepths, haveNumDepths = self.__parse_val("number of depth levels")
        if not haveNumDepths: 
            warn('Number of Depth levels not defined.')
            self.numDepths = 6
        self.numDepths = int(self.numDepths)
        # get balloon matrices
        self.__init_matrices()
        self.__parse_matrix('V0', 0, self.V0)[0]
        self.__parse_matrix('F0', 0, self.F0)[0]
        self.__parse_matrix('tau0', 0, self.tau0)[0]
        self.__parse_matrix('alpha', 0, self.alpha)[0]
        self.__parse_matrix('visco-elastic time constants', -1, self.vet)[0]
        # get BOLD-parameters
        numBOLDparams = int(self.__parse_val("number of BOLD parameters")[0])
        boldparams_tmp = readMatrixFromText(self.filetext, 'BOLD', 0, self.numCompartments, numBOLDparams)[0]
        self.E0 = boldparams_tmp[:,0]
        self.B0 = self.__parse_val("B0")[0]
        self.n = self.__parse_val("n-ratio")[0]
        self.boldparams = {
            'epsilon': boldparams_tmp[:,1],
            'Hct': boldparams_tmp[:,2],
            'r0': boldparams_tmp[:,3],
            'dXi': self.__parse_val("dXi")[0],
            'TE': self.__parse_val("TE")[0],
            'gamma0': 2*np.pi*42.58*pow(10,6) }

# ---------------------------------  CHECK IF INPUT IS VIABLE  -------------------------------- 
    ''' __countChangedVals: count, how many inserted values are on one layer '''
    def __countChangedVals(self, matrix, initVal):
        return sum(matrix != initVal*np.ones(np.size(matrix)))

    ''' __inputCheckFailed: raise exception, if input is not viable '''
    def __inputCheckFailed(self, under, over):
        if under > 0: errorcase = f'under-determined on {under} layers'
        if over > 0: errorcase = f'over-determined on {over} layers'
        if under > 0 and over > 0: 
            errorcase = \
            f'under-determined on {under} layers and over-determined on {over} layers'
        errormessage = \
            'Given input for F0, V0, tau0 is ' + errorcase + '. \n' + \
            'Input should have the following form on each layer: \n' + \
            '    - 1 value for V0 or tau0 in venule compartment \n' + \
            '    - 1 value for V0 or tau0 in venous compartment \n' + \
            '    - a third value that can be (either/ or): \n' + \
            '           * 1 value for F0 (in any compartment) \n' + \
            '           * an additional value for V0 or tau0 in venous or venule compartment \n'
        raise Exception(errormessage)

    ''' __checkInput: make sure F0, V0, tau0 meet requirements for full rank definition 
            * raise exception if not
            * required format described in: inputCheckFailed -> errormessage
        OUTPUT: haveF0 -> array[numDepths,2] saves for each depth, if have a value for F0 
                                             or the compartment where it can be calculated '''
    def __checkInput(self):
        # init values
        under = 0
        over = 0
        haveF0 = np.zeros([self.numDepths, 2])  # row1: haveF0; row2: compartment with 2 values (if not haveF0)
        valsPerLayer = self.numCompartments
        # go through all layers
        for d in range(0, self.numDepths):
            # get number of elements
            sF0 = self.__countChangedVals(self.F0[:,d], self.__initVals['F0'])
            sV0 = self.__countChangedVals(\
                self.V0[self.VENULE:self.numCompartments,d], self.__initVals['V0'])
            st0 = self.__countChangedVals(\
                self.tau0[self.VENULE:self.numCompartments,d], self.__initVals['tau0'])
            # check if error
            if sF0 + sV0 + st0 != valsPerLayer or sF0 > 1:
                under += sF0 + sV0 + st0 < valsPerLayer
                over += sF0 + sV0 + st0 > valsPerLayer or sF0 > 1
                continue
            # save if have F0
            haveF0[d,0] = sF0 == 1
            # check that at least one value for venule, vein compartments each
            for k in range(self.VENULE, self.numCompartments):
                change = int(self.V0[k,d] != self.__initVals['V0']) \
                       + int(self.tau0[k,d] != self.__initVals['tau0'])
                under += int(change==0)
                # save compartment where F0 can be calculated
                if not haveF0[d,0] and change == 2:
                    haveF0[d,1] = int(k)
        # throw exception if any layer is faulty
        if not under + over == 0:
            self.__inputCheckFailed(under, over)
        return haveF0

# ----------------------------------  FILL V0, F0, TAU0  ---------------------------------------
    ''' __getThirdValueVFt: get one value for {tau0 = V0/F0} '''
    def __getThirdValueVFt(self, k,d):
        f = self.F0[k,d] != self.__initVals['F0']
        v = self.V0[k,d] != self.__initVals['V0']
        t = self.tau0[k,d] != self.__initVals['tau0']
        if not v and f and t: self.V0[k,d] = self.F0[k,d] * self.tau0[k,d]
        elif not f and v and t: self.F0[k,d] = self.V0[k,d] / self.tau0[k,d]
        elif not t and v and f: self.tau0[k,d] = self.V0[k,d] / self.F0[k,d]
        
    ''' __fillVFt: get all values for V0, F0, tau0 (assume requirements are met) '''
    def __fillVFt(self, haveF0):
        for d in range(self.numDepths-1, -1, -1):
            # make sure there is exactly one column of F0
            if not haveF0[d,0]:
                k = int(haveF0[d,1])
                self.__getThirdValueVFt(k,d)
            # fill F0
            self.__makeFlowMeetConditions(d)
            # tau0 = V0/F0
            for k in range(self.VENULE, self.numCompartments):
                self.__getThirdValueVFt(k,d)
    
    ''' __completeInput: check if V0,F0,tau0 meet requirements and calculate missing values '''
    def __completeInput(self):
        haveF0 = self.__checkInput()
        self.__fillVFt(haveF0)

# -------------------------------------  GET F0  ----------------------------------------------
    ''' __listIncludesCompartment: return, whether a specific compartment is included in a list '''
    def __listIncludesCompartment(self, k_list, compartment):
        return any(k == compartment for k in k_list)
    
    ''' __needDeeperLayerFormula: F0 resting state condition requires deeper layer for veins (except deepest layer) '''
    def __needDeeperLayerFormula(self, k_list, d):
        return self.__listIncludesCompartment(k_list, self.VEIN) and d<self.numDepths-1
    
    ''' __findChangedVal: return first compartment of list that was changed since init of matrix '''
    def __findChangedVal(self, k_list, d):
        for k in k_list:
            if self.F0[k,d] != self.__initVals['F0']: return k
    
    ''' __makeFlowMeetCondition_oneCompartment: make flow in one compartment meet resting state conditions '''
    def __makeFlowMeetCondition_oneCompartment(self, k, k_groundTruth, d):
        if not self.__needDeeperLayerFormula([k, k_groundTruth], d):
            self.F0[k,d] = self.F0[k_groundTruth,d]
        else:  
            if k_groundTruth == self.VEIN:
                self.F0[k,d] = self.F0[self.VEIN,d] - self.F0[self.VEIN,d+1]
            else:
                self.F0[k,d] = self.F0[k_groundTruth,d] + self.F0[self.VEIN,d+1]  
    
    ''' __makeFlowMeetConditions: make flow in all compartments meet resting state conditions '''
    def __makeFlowMeetConditions(self, d):
        k_list = [self.ARTERIOLE, self.VENULE, self.VEIN]
        k_groundTruth = self.__findChangedVal(k_list, d)
        for k in range(0, self.numCompartments):
            if k==k_groundTruth: continue
            self.__makeFlowMeetCondition_oneCompartment(k, k_groundTruth, d)
