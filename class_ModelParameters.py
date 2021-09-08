
"""
@name:      Model_Parameters
@author:    Sula Spiegel
@change:    05/08/2021

@summary:   Class to store general model parameters (read them from file first)
"""

import numpy as np
from warnUsr import warn

from numpy.core.fromnumeric import ndim
from readFile import getFileText, readValFromText, readMatrixFromText

class Model_Parameters:
    def __init__(self, parameter_file: str):
        self.parameter_file = parameter_file
        self.COMPARTMENTS = {"ARTERIOLE", "VENULE", "VEIN"}
        self.ARTERIOLE = 0
        self.VENULE = 1
        self.VEIN = 2
        self.INFLATION = 0  # for visco-elastic time constant
        self.DEFLATION = 1

        self.readindex = {  # indice of readableVars
            'readname': 0,
            'defaultVal': 1,
            'throwException': 2,
            'type': 3,
            'nVar': 3
        }
        self.readableVars = {  # attrname: [name for read-in, defaultVal, throwException, type/nVar]
            'singular': {
                'N' : ["number of time points", -1, True, 'int'],
                'dt' : ["time integration step (dt)", 0.01, False, 'float'],
                'numDepths' : ["number of depth levels", -1, False, 'int'],
            },
            'matrices' : {
                'V0' : ['V0', 0, False, 0],
                'F0' : ['F0', 0, False, 0],
                'tau0' : ['tau0', 0, False, 0],
                'alpha' : ['alpha', 1, True, 0],
                'vet' : ['visco-elastic time constants', 1, True, -1],
                'E0' : ['E0', 1, False, 0]
            },
            'BOLDvars' : { 
                'singular': {
                    'B0' : ['B0', -1, False, 'float'],
                    'dXi': ['dXi', -1, False, 'float'], 
                    'TE': ['TE', -1, False, 'float']
                },
                'matrices' : {
                    'epsilon' : ['epsilon', -1, False, 0],
                    'Hct' : ['Hct', -1, False, 0],
                    'r0' : ['r0', -1, False, 0],
                }
            }
        }

        self.haveBOLDparams = True
        self.numCompartments = len(self.COMPARTMENTS)
        self.__parse_parameterFile(parameter_file)
        self.__completeInput()
        
# ---------------------------------  EXPLICIT READ-IN  ----------------------------------------
    ''' __addNewException: add a new line of errormessage if a required value is missing '''
    def __addNewException(self, readname):
        self._exception = self._exception + f'\n{readname} missing. Define it in {self.parameter_file}.'
    
    ''' __checkIfHaveMatrixDim: check, if the specific dimension of matrix is already given (numComp/numDepths) '''
    def __checkIfHaveMatrixDim(self, attrname):
        if getattr(self, attrname, None) is None: 
            setattr(self, attrname, -1)
            return False
        return getattr(self, attrname) != -1
    ''' __checkIfDimsMatch: check, if dimensions of two read-in matrices match '''
    def __checkIfDimsMatch(self, mat, dimnum, haveDim, oldVal, attrname):
        if mat is not None and not haveDim:
            newval = mat.shape[dimnum]
            if newval == oldVal:
                setattr(self, attrname, int(oldVal))
                haveDim = True
            else: oldVal = newval
        return haveDim, oldVal
    ''' __getBasicMatrixDims: make sure, self.numDepths and self.numCompartments are defined '''
    def __getBasicMatrixDims(self):
        # check if getting dims is necessary
        dims = ['numCompartments', 'numDepths']
        haveDim = [self.__checkIfHaveMatrixDim(x) for x in dims]
        if all(haveDim): return
        # get dims
        oldVal = (-1) * np.ones([2])
        for matname in self.readableVars['matrices'].keys():
            readname = self.readableVars['matrices'][matname][self.readindex['readname']]
            nnVar = self.readableVars['matrices'][matname][self.readindex['nVar']]
            mat = readMatrixFromText(self.filetext, readname, nVar=nnVar)
            for i in range(0, len(dims)):
                haveDim[i], oldVal[i] = self.__checkIfDimsMatch(mat, i, haveDim[i], oldVal[i], dims[i])
            if all(haveDim): return
        raise Exception('\
            \nNumber of Depth levels not defined. \
            \nNeeded as direct input ("number of depth levels = ...") \
            \nor can be derived from resting state conditions.\n')

    ''' __init_matrices: initialize all matrices needed '''
    def __init_matrices(self):
        for matname in self.readableVars['matrices'].keys():  # ['V0', 'F0', 'tau0', 'alpha', 'vet', 'E0']
            if self.readableVars['matrices'][matname][self.readindex['nVar']] > -1:
                initVal = self.readableVars['matrices'][matname][self.readindex['defaultVal']]
                setattr(self, matname, np.ones([self.numCompartments, self.numDepths]) * initVal)

    ''' __parse: read a single value or matrix (vartype = 'singular'/'matrices') from the parameter file '''
    def __parse(self, vartype, varname, bold):
        if not bold: readable = self.readableVars
        else: readable = self.readableVars['BOLDvars']
        readname = readable[vartype][varname][self.readindex['readname']]
        wouldThrow = readable[vartype][varname][self.readindex['throwException']]
        if vartype == 'singular': 
            typestring = readable[vartype][varname][self.readindex['type']]
            res = readValFromText(self.filetext, readname, typestring)
            if res is None:
                res = readable[vartype][varname][self.readindex['defaultVal']]
                if not wouldThrow and not bold: warn(f'{readname} not given. Using defaultVal {varname} = {res}')
        if vartype == 'matrices':
            res = getattr(self, varname, None)
            nVar = readable[vartype][varname][self.readindex['nVar']]
            res = readMatrixFromText(self.filetext, readname, self.numCompartments, self.numDepths, nVar, res)
        if not bold: setattr(self, varname, res)
        else: self.boldparams[varname] = res
        if wouldThrow and self.__isInit(varname, vartype, bold): self.__addNewException(readname)
        return res
    ''' __parse_val: read a single value from the parameter file ''' 
    def __parse_val(self, varname, bold=False):
        self.__parse('singular', varname, bold)
    ''' __parse_matrix: read an entire matrix from the parameter file '''
    def __parse_matrix(self, varname, bold=False):
        self.__parse('matrices', varname, bold)
    
    ''' __isInit: return True, if a matrix/value is in initial conditions (False if not) '''
    def __isInit(self, varname, vartype='matrices', bold=False):
        if not bold: 
            readable = self.readableVars
            s = self
        else: 
            readable = self.readableVars['BOLDvars']
            s = self.boldparams
        var = getattr(s, varname, None)
        if var is None: return True
        initVal = readable[vartype][varname][self.readindex['defaultVal']]
        if vartype == 'singular': return var == initVal
        else: return (var==initVal).all()
        #sum(var - initVal*np.ones(np.size(var)) == 0)
    
    ''' __BOLDcheck: check, that all BOLD-parameters are there '''
    def __BOLDcheck(self, vartype):
        for varname in self.readableVars['BOLDvars'][vartype].keys():
            if self.__isInit(varname, vartype, bold=True):
                readname = self.readableVars['BOLDvars'][vartype][varname][self.readindex['readname']]
                warn(f'{readname} not given. Won\'t calculate BOLD-contrast.')
                self.haveBOLDparams = False

    ''' __parse_parameterFile: read all required parameters from the parameter file '''
    def __parse_parameterFile(self, filename):
        self._exception = ''  # throw exception if values are missing
        self.filetext = getFileText(filename)  # get total file as string
        # get singular values
        for valuename in self.readableVars['singular'].keys(): self.__parse_val(valuename)
        # get balloon matrices
        self.__getBasicMatrixDims()
        self.__init_matrices()  # need init, so that they have correct initVal
        for matname in self.readableVars['matrices'].keys(): self.__parse_matrix(matname)
        # check init conditions (need 2 out of 3 -> can't just use general exception) 
        if sum(not self.__isInit(x) for x in ['V0', 'F0', 'tau0']) < 2: 
            self._exception = self._exception + \
                '\nResting conditions required. Define three columns of F0, V0 or tau0 (see comments for specific format). '
        # get BOLD-parameters
        self.boldparams = { 'gamma0': 2*np.pi*42.58*pow(10,6) }
        boldparams_tmp, _, titles = readMatrixFromText(
            self.filetext, 'BOLD', self.numCompartments, numDepths=-1, nVar=0, needCaptions=True)
        if boldparams_tmp is None: warn(f'No BOLD parameters given. Won\'t calculate BOLD-contrast.')
        elif titles is None: warn(f'Names of BOLD-parameters not clear. Won\'t calculate BOLD-contrast.')
        else:
            for i in range(0, len(titles)): self.boldparams[titles[i]] = boldparams_tmp[:,i]
            for valuename in self.readableVars['BOLDvars']['singular'].keys(): self.__parse_val(valuename, bold=True)
            # check, if BOLD-params are complete (and warn if not, but don't throw)
            for vartype in ['singular', 'matrices']: self.__BOLDcheck(vartype)
        # throw exception if anything is missing
        if len(self._exception) > 0: raise Exception(self._exception + '\n\n')

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
        initInd = self.readindex['defaultVal']
        # go through all layers
        for d in range(0, self.numDepths):
            # get number of elements
            sF0 = self.__countChangedVals(self.F0[:,d], self.readableVars['matrices']['F0'][initInd])
            sV0 = self.__countChangedVals(\
                self.V0[self.VENULE:self.numCompartments,d], self.readableVars['matrices']['V0'][initInd])
            st0 = self.__countChangedVals(\
                self.tau0[self.VENULE:self.numCompartments,d], self.readableVars['matrices']['tau0'][initInd])
            # check if error
            if sF0 + sV0 + st0 != valsPerLayer or sF0 > 1:
                under += sF0 + sV0 + st0 < valsPerLayer
                over += sF0 + sV0 + st0 > valsPerLayer or sF0 > 1
                continue
            # save if have F0
            haveF0[d,0] = sF0 == 1
            # check that at least one value for venule, vein compartments each
            for k in range(self.VENULE, self.numCompartments):
                change = int(self.V0[k,d] != self.readableVars['matrices']['V0'][initInd]) \
                       + int(self.tau0[k,d] != self.readableVars['matrices']['tau0'][initInd])
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
        f = self.F0[k,d] != self.readableVars['matrices']['F0'][self.readindex['defaultVal']]
        v = self.V0[k,d] != self.readableVars['matrices']['V0'][self.readindex['defaultVal']]
        t = self.tau0[k,d] != self.readableVars['matrices']['tau0'][self.readindex['defaultVal']]
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
            if self.F0[k,d] != self.readableVars['matrices']['F0'][self.readindex['defaultVal']]: 
                return k
    
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
