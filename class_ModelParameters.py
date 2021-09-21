
"""
@name:      Model_Parameters
@author:    Sula Spiegel
@change:    05/08/2021

@summary:   Class to store general model parameters (read them from file first)
"""

import numpy as np
from numpy.core.fromnumeric import ndim

from warnUsr import warn
from readFile import getFileText, readValFromText, readMatrixFromText

''' makeDict: create a dictionary with specified entries '''
def makeDict(entrynames, values=None):
    d = {}
    for i in range(0, len(entrynames)): 
        if values is not None: d[entrynames[i]] = values[i]
        else: d[entrynames[i]] = i
    return d

''' clearAttrs: delete all attrs of attrlist '''
def clearAttrs(obj, attrlist):
    for attr in attrlist: delattr(obj, attr)

''' readableVarNameClass: define structure of "readableVars" in Model_Parameters '''
class readableVarNameClass:
    def __init__(self):
        self.readInfo = ['readname', 'defaultVal','throwException','type','nVar']  # indice of readableVars 
        self.__setSelfStrings(self.readInfo)  # self.readname = 'readname'
        setattr(self, 'readindex', makeDict(self.readInfo))  # self.readindex['readname'] = 0
        self.readindex['nVar'] = 3  # 'nVar' and 'type' share index=3

        self.vartypes = ['single', 'matrix', 'BOLDvars']
        self.__setSelfStrings(self.vartypes)  # self.single = 'single'
    
    ''' __setSelfStrings: give each string in self.headname it's own name as value ( eg self.readname = 'readname' ) '''
    def __setSelfStrings(self, head):
        for i in range(0, len(head)): setattr(self, head[i], head[i])  

nameClass = readableVarNameClass()  # initialize instance

class readableVarClass:
    def __init__(self):
        self.readableVars = {  # attrname: [name for read-in, defaultVal, throwException, type/nVar]
            nameClass.single : {
                'N' : ["number of time points", -1, True, 'int'],
                'dt' : ["time integration step (dt)", 0.01, False, 'float'],
                'numDepths' : ["number of depth levels", -1, False, 'int'],
            },
            nameClass.matrix : {
                'V0' : ['V0', 0, False, 0],
                'F0' : ['F0', 0, False, 0],
                'tau0' : ['tau0', 0, False, 0],
                'alpha' : ['alpha', 1, True, 0],
                'vet' : ['visco-elastic time constants', 1, True, -1],
                'E0' : ['E0', 1, False, 0],
                'n' : ['n-ratio', 1, False, 0]
            },
            nameClass.BOLDvars : { 
                nameClass.single : {
                    'B0' : ['B0', -1, False, 'float'],
                    'dXi': ['dXi', -1, False, 'float'], 
                    'TE': ['TE', -1, False, 'float']
                },
                nameClass.matrix : {
                    'epsilon' : ['epsilon', -1, False, 0],
                    'Hct' : ['Hct', -1, False, 0],
                    'r0' : ['r0', -1, False, 0],
                    'E0' : ['E0', -1, False, 0]
                }
            }
        }

class Model_Parameters:
    def __init__(self, parameter_file: str):
        self.COMPARTMENTS = ["ARTERIOLE", "VENULE", "VEIN"]
        self.DIMS = ['numCompartments', 'numDepths']
        self.FLOWDIRS = ["INFLATION", "DEFLATION"]  # for visco-elastic time constant
        self._OXMODES = ["_OX_n", "_OX_m", "_OX_E0"]

        # give each entry in COMPARTMENTS, DIMS and FLOWDIRS an attribute and value
        self.__setAllCONSTS()  # eg self.ARTERIOLE = 0 etc

        self.parameter_file = parameter_file
        self.nameClass = readableVarNameClass()
        self.readableVars = readableVarClass().readableVars
        
        self.haveBOLDparams = True
        self.numCompartments = len(self.COMPARTMENTS)

        self.__parse_parameterFile(parameter_file)  # read parameters from file
        self.__completeVFt()  # make sure, V,F and tau meet requirements for resting conditions
        self.__checkQ()  # make sure, given input is sufficient to calculate ox-extraction (q, dq)
        clearAttrs(self, ['filetext', 'haveBOLDparams'])

# --------------------------------------  HELPERS  --------------------------------------------
    ''' __setAllCONSTS: give each string in attributes given so far a value ( eg self.ARTERIOLE = 0 ) '''
    def __setAllCONSTS(self):
        # get all attributes defined so far
        for headname in [x for x in [i for i in self.__dict__.keys()]]:
            head = getattr(self, headname)
            # set each individual string as attribute with index as value
            for i in range(0, len(head)): setattr(self, head[i], i)

    def __getReadable(self, bold=False):
        if not bold: return self.readableVars
        else: return self.readableVars[nameClass.BOLDvars]
    
    ''' __getAllValuenames: get all "readableVars" (names of variables) of specific vartype '''
    def __getAllValuenames(self, vartype, bold=False):
        return self.__getReadable(bold)[vartype].keys()
    
    ''' getVarInfo: get entry from "readableVars" 
            INPUT:  * varname: attrname (eg 'V0')
                    * vartype: 'single' or 'matrix'
                    * readinfo: required info ('readname', 'defaultVal','throwException','type' or 'nVar') '''
    def getVarInfo(self, varname, vartype, readinfo, bold=False):
        return self.__getReadable(bold)[vartype][varname][nameClass.readindex[readinfo]]
    
    ''' __getInitVal: return initial/ default val for a varname '''
    def __getInitVal(self, varname, vartype=nameClass.matrix, bold=False):
        return self.getVarInfo(varname, vartype, nameClass.defaultVal, bold)

# ---------------------------------  EXPLICIT READ-IN  ----------------------------------------
    ''' __addNewException: add a new line of errormessage if a required value is missing '''
    def __addNewException(self, readname):
        self._exception = self._exception + f'\n{readname} missing. Define it in {self.parameter_file}.'
    
    ''' __checkIfHaveMatrixDim: check, if the specific dimension of matrix is already given (numComp/numDepths) '''
    def __checkIfHaveMatrixDim(self, dimname):
        if getattr(self, dimname, None) is None: 
            setattr(self, dimname, -1)
            return False
        return getattr(self, dimname) != -1

    ''' __checkIfDimsMatch: check, if dimensions of two read-in matrices match '''
    def __checkIfDimsMatch(self, mat, dimnum, haveDim, oldVal):
        if mat is not None and not haveDim[dimnum]:
            newval = mat.shape[dimnum]
            if newval == oldVal[dimnum]:
                setattr(self, self.DIMS[dimnum], int(newval))
                haveDim[dimnum] = True
            else: oldVal[dimnum] = newval
        return haveDim, oldVal

    ''' __getBasicMatrixDims: make sure, self.numDepths and self.numCompartments are defined '''
    def __getBasicMatrixDims(self):
        # check if getting dims is necessary
        haveDim = [self.__checkIfHaveMatrixDim(x) for x in self.DIMS]
        if all(haveDim): return
        # get dims
        oldVal = (-1) * np.ones([len(self.DIMS)])
        for matname in self.__getAllValuenames(nameClass.matrix):
            readname = self.getVarInfo(matname, nameClass.matrix, nameClass.readname)
            nVar = self.getVarInfo(matname, nameClass.matrix, nameClass.nVar)
            mat = readMatrixFromText(self.filetext, readname, nVar)
            for i in range(0, len(self.DIMS)):
                haveDim, oldVal = self.__checkIfDimsMatch(mat, i, haveDim, oldVal)
            if all(haveDim): return
        raise Exception('\
            \nNumber of Depth levels not defined. \
            \nNeeded as direct input ("number of depth levels = ...") \
            \nor can be derived from resting state conditions.\n')

    ''' __init_matrices: initialize all matrices needed '''
    def __init_matrices(self):
        for matname in self.__getAllValuenames(nameClass.matrix):
            if self.getVarInfo(matname, nameClass.matrix, nameClass.nVar) > -1:
                initVal = self.__getInitVal(matname)
                setattr(self, matname, np.ones([self.numCompartments, self.numDepths]) * initVal)
    
    ''' __isInit: return False, if a matrix/value is not in initial conditions any more
                  return True, if is in initial conditions or hasn't been initialized yet '''
    def __isInit(self, varname, vartype=nameClass.matrix, bold=False):
        if not bold: var = getattr(self, varname, None)
        else: var = self.boldparams[varname]
        if var is None: return True
        initVal = self.__getInitVal(varname, vartype, bold)
        if vartype == nameClass.single: return var == initVal
        else: return (var==initVal).all()

    ''' __parse: read a single value or matrix (vartype = 'single'/'matrix') from the parameter file '''
    def __parse(self, varname, vartype=nameClass.matrix, bold=False):
        readname = self.getVarInfo(varname, vartype, nameClass.readname, bold)
        wouldThrow = self.getVarInfo(varname, vartype, nameClass.throwException, bold)
        if vartype == nameClass.single: 
            typestring = self.getVarInfo(varname, vartype, nameClass.type, bold)
            res = readValFromText(self.filetext, readname, typestring)
            if res is None:
                res = self.getVarInfo(varname, vartype, nameClass.defaultVal, bold)
                if not wouldThrow and not bold: warn(f'{readname} not given. Using defaultVal {varname} = {res}')
        if vartype == nameClass.matrix:
            res = getattr(self, varname, None)
            nVar = self.getVarInfo(varname, vartype, nameClass.nVar, bold)
            res = readMatrixFromText(self.filetext, readname, self.numCompartments, self.numDepths, nVar, res)
        if not bold: setattr(self, varname, res)
        else: self.boldparams[varname] = res
        if wouldThrow and self.__isInit(varname, vartype, bold): self.__addNewException(readname)
        return res

    ''' __E0check: E0 can be given as part of boldparams or independently -> make sure at least one is given. '''
    def __E0check(self):
        if self.__checkE0: return True  # if E0 was given as matrix
        return hasattr(self.boldparams, 'E0')

    ''' __BOLDfail: define what to do if a bold-param is missing '''
    def __BOLDfail(self, varname, vartype):
        readname = self.getVarInfo(varname, vartype, nameClass.readname, bold=True)
        warn(f'{readname} not given. Won\'t calculate BOLD-contrast.')
        self.haveBOLDparams = False
    
    ''' __BOLDcheck: check, that all BOLD-parameters are there '''
    def __BOLDcheck(self, vartype):
        for varname in self.__getAllValuenames(vartype, bold=True):
            if self.__isInit(varname, vartype, bold=True):
                if varname == 'E0': 
                    if self.__E0check(): continue
                self.__BOLDfail(varname, vartype)
                    
    ''' __parse_parameterFile: read all required parameters from the parameter file '''
    def __parse_parameterFile(self, filename):
        self._exception = ''  # throw exception if values are missing
        self.filetext = getFileText(filename)  # get total file as string
        # get singular values
        for valuename in self.__getAllValuenames(nameClass.single): self.__parse(valuename, nameClass.single)
        # get balloon matrices
        self.__getBasicMatrixDims()
        self.__init_matrices()  # need init, so that they have correct initVal
        for matname in self.__getAllValuenames(nameClass.matrix): self.__parse(matname)
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
            for valuename in self.__getAllValuenames(nameClass.single, bold=True): 
                self.__parse(valuename, nameClass.single, bold=True)
            # check, if BOLD-params are complete (and warn if not, but don't throw)
            for vartype in nameClass.vartypes[0:1]: self.__BOLDcheck(vartype)
        # throw exception if anything is missing
        if len(self._exception) > 0: raise Exception(self._exception + '\n\n')

# ---------------------------------  CHECK IF INPUT IS VIABLE  -------------------------------- 
    ''' __countChangedVals: count, how many inserted values are on one layer '''
    def __countChangedVals(self, matrix, varname):
        return sum(matrix != self.__getInitVal(varname) * np.ones(np.size(matrix)))

    ''' __throwIfFailed: raise exception, if input is not viable '''
    def __throwIfFailed(self, under, over):
        errormsg = ''
        if under > 0: errormsg = errormsg + f'under-determined on {under} layer'
        if under > 1: errormsg = errormsg + 's'
        if under > 0 and over > 0: errormsg = errormsg + ' and '
        if over > 0: errormsg = errormsg + f'over-determined on {over} layer'
        if over > 1: errormsg = errormsg + 's'
        if len(errormsg) > 0: 
            errormsg = \
            'Given input for F0, V0, tau0 is ' + errormsg + '. \n' + \
            'Input should have the following form on each layer: \n' + \
            '    - 1 value in venule compartment (V0 or tau0) \n' + \
            '    - 1 value in venous compartment (V0 or tau0) \n' + \
            '    - a third value that can be (either/ or): \n' + \
            '           * 1 value for F0 (in any compartment) \n' + \
            '           * an additional value for V0 or tau0 in venous or venule compartment \n'
            raise Exception(errormsg)

    ''' __checkInput: make sure F0, V0, tau0 meet requirements for full rank definition 
            * raise exception if not
            * required format described in: inputCheckFailed -> errormessage
        OUTPUT: haveF0 -> array[numDepths,2]; saves for each depth, if have a value for F0 
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
            sF0 = self.__countChangedVals(self.F0[:,d], 'F0')
            sV0 = self.__countChangedVals(self.V0[self.VENULE:self.numCompartments,d], 'V0')
            st0 = self.__countChangedVals(self.tau0[self.VENULE:self.numCompartments,d], 'tau0')
            # check if error
            if sF0 + sV0 + st0 != valsPerLayer or sF0 > 1:
                under += sF0 + sV0 + st0 < valsPerLayer
                over += sF0 + sV0 + st0 > valsPerLayer or sF0 > 1
                continue
            # save if have F0
            haveF0[d,0] = sF0 == 1
            # check that at least one value for venule, vein compartments each
            for k in range(self.VENULE, self.numCompartments):
                change = \
                      int(self.V0[k,d] != self.__getInitVal('V0')) \
                    + int(self.tau0[k,d] != self.__getInitVal('tau0'))
                under += int(change==0)
                # save compartment where F0 can be calculated
                if not haveF0[d,0] and change == 2:
                    haveF0[d,1] = int(k)
        # throw exception if any layer is faulty
        self.__throwIfFailed(under, over)
        return haveF0

# ----------------------------------  FILL V0, F0, TAU0  ---------------------------------------
    ''' __getThirdValueVFt: get one value for {tau0 = V0/F0} '''
    def __getThirdValueVFt(self, k,d):
        f = self.F0[k,d] != self.__getInitVal('F0')
        v = self.V0[k,d] != self.__getInitVal('V0')
        t = self.tau0[k,d] != self.__getInitVal('tau0')
        if not v and f and t: self.V0[k,d] = self.F0[k,d] * self.tau0[k,d]
        if not f and v and t: self.F0[k,d] = self.V0[k,d] / self.tau0[k,d]
        if not t and v and f: self.tau0[k,d] = self.V0[k,d] / self.F0[k,d]
        
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
    
    ''' __completeVFt: check if V0,F0,tau0 meet requirements and calculate missing values '''
    def __completeVFt(self):
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
            if self.F0[k,d] != self.__getInitVal('F0'): 
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

# -----------------------------  MAKE SURE Q CAN BE CALCULATED  -------------------------------
    ''' __checkE0: make sure, E0 was given as matrix and not as part of boldparams '''
    def __checkE0(self):
        if self.__isInit('E0'): return False
        if (self.E0>0.7).any() or (self.E0<0).any(): return False
        for attr in self.boldparams:
            if attr == 'E0': continue
            for d in range(0, self.numDepths):
                if (self.E0[:,d]==self.boldparams[attr]).all(): return False
        return True
    
    ''' __checkQ: make sure, given input is sufficient to calculate ox-extraction (q, dq) '''
    def __checkQ(self):
        if not self.__isInit('n'): self.oxmode = self._OX_n
        elif self.__checkE0(): self.oxmode = self._OX_E0
        else: 
            self.oxmode = self._OX_m
            self._exception = (f"\ndHb-content can not be calculated with given input. "
                               f"\nGive either "
                               f"{self.getVarInfo('n', nameClass.matrix, nameClass.readname)} or "
                               f"{self.getVarInfo('E0', nameClass.matrix, nameClass.readname)} "
                               f"as matrix [{self.DIMS[0]}, {self.DIMS[1]}].\n"
                               f"Alternatively, give CMRO2 as inputfile into 'Input_Timeline'.\n\n")
