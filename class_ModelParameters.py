
"""
@name:      Model_Parameters
@author:    Sula Spiegel
@change:    05/08/2021

@summary:   Class to store general model parameters (read them from file first)
"""

from hashlib import new
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

''' clearAttrs: delete all attrs in attrlist from obj '''
def clearAttrs(obj, attrlist):
    for attr in attrlist: delattr(obj, attr)

''' readableVarNameClass: define structure of readableVarClass (see below) '''
class readableVarNameClass:
    def __init__(self):
        self.readInfo = ['readname','defaultVal','throwException','changeable','type','nVar']  # necessary info to create a readableVar
        self.__setSelfStrings(self.readInfo)  # self.readname = 'readname' for all strings in readInfo
        setattr(self, 'readindex', makeDict(self.readInfo))  # indice of info in readableVar ( self.readindex['readname'] = 0 )
        self.readindex['nVar'] = len(self.readInfo)-2  # 'nVar' and 'type' share index (depending on whether single/matrix (matrix is always float))

        self.vartypes = ['single', 'matrix', 'BOLDvars']
        self.__setSelfStrings(self.vartypes)  # self.single = 'single'
    
    ''' __setSelfStrings: give each string in self.headname it's own name as value ( eg self.readname = 'readname' ) '''
    def __setSelfStrings(self, head):
        for i in range(0, len(head)): setattr(self, head[i], head[i])  

nameClass = readableVarNameClass()  # initialize instance

''' readableVarClass: class to define variables that can be used in Model_Parameters 
                      (assume they are read in via file) '''
class readableVarClass:
    def __init__(self):
        self.readableVars = {  # attrname: [name for read-in, defaultVal, throw exception if missing, changeable, type/nVar]
            nameClass.single : {
                'N' : ["number of time points", -1, True, False, 'int'],
                'dt' : ["time integration step (dt)", 0.01, False, True, 'float'],
                'numDepths' : ["number of depth levels", -1, False, False, 'int'],
            },
            nameClass.matrix : {
                'V0' : ['V0', 0, False, True, 0],
                'F0' : ['F0', 0, False, True, 0],
                'tau0' : ['tau0', 0, False, True, 0],
                'alpha' : ['alpha', 1, True, True, 0],
                'vet' : ['visco-elastic time constants', 1, True, True, -1],
                'E0' : ['E0', 1, False, True, 0],
                'n' : ['n-ratio', 1, False, True, 0]
            },
            nameClass.BOLDvars : { 
                nameClass.single : {
                    'B0' : ['B0', -1, False, True, 'float'],
                    'dXi': ['dXi', -1, False, True, 'float'], 
                    'TE': ['TE_', -1, False, True, 'float']
                },
                nameClass.matrix : {
                    'epsilon' : ['epsilon', -1, False, True, 0],
                    'Hct' : ['Hct', -1, False, True, 0],
                    'r0' : ['r0', -1, False, True, 0],
                    'E0' : ['E0', -1, False, True, 0]
                }
            }
        }

''' Model_Parameters: '''
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

# --------------------------------------  GET-FUNCTIONS  --------------------------------------------
    ''' __getReadable: get part of "readableVars" that is bold/not bold (needed for __getAllValuenames) '''
    def __getReadable(self, bold=False):
        if not bold: return self.readableVars
        else: return self.readableVars[nameClass.BOLDvars]
    
    ''' __getAllValuenames: get names of all "readableVars" of a specific vartype (single/matrix) '''
    def __getAllValuenames(self, vartype, bold=False):
        return self.__getReadable(bold)[vartype].keys()
    
    ''' getVarInfo: get entry from "readableVars" 
            INPUT:  * varname: attrname (eg 'V0')
                    * vartype: 'single' or 'matrix'
                    * readinfo: required info ('readname', 'defaultVal','throwException','type' or 'nVar') '''
    def getVarInfo(self, varname, vartype, readinfo, bold=False):
        return self.__getReadable(bold)[vartype][varname][nameClass.readindex[readinfo]]
    
    ''' getVarValue: returns the value of a specific variable by varname '''
    def getVarValue(self, varname, bold=False):
        if not bold: return getattr(self, varname, None)
        else: return self.boldparams[varname]
    
    ''' __getInitVal: return initial/ default val for a varname '''
    def __getInitVal(self, varname, vartype=nameClass.matrix, bold=False):
        return self.getVarInfo(varname, vartype, nameClass.defaultVal, bold)
    
    ''' findVarname: find out, where a variable <varname> is stored (boldparams or not + single/matrix) '''
    def findVarname(self, varname):
        for vartype in [nameClass.single, nameClass.matrix]:
            for b in [False, True]:
                if varname in self.__getAllValuenames(vartype, bold=b):
                    return vartype, b
        return None, None

# --------------------------------------  SET-FUNCTIONS  ----------------------------------------
    ''' __setAllCONSTS: give each string in attributes given so far a value ( eg self.ARTERIOLE = 0 ) '''
    def __setAllCONSTS(self):
        # get all attributes defined so far
        for headname in [x for x in [i for i in self.__dict__.keys()]]:
            head = getattr(self, headname)
            # set each individual string as attribute with index as value
            for i in range(0, len(head)): setattr(self, head[i], i)
    
    ''' __setVar: set a variable as attribute of self '''
    def __setVar(self, varname, val, bold=False):
        if not bold: setattr(self, varname, val)
        else: self.boldparams[varname] = val
    

# ---------------------------------  READ-IN FROM FILE ----------------------------------------
    ''' __addNewException: add a new line of errormessage if a required value is missing '''
    def __addNewException(self, readname):
        self._exception = self._exception + f'\n{readname} missing. Define it in {self.parameter_file}.'
    
    ''' __checkIfHaveMatrixDim: check, if the specific dimension of matrix is already given (numComp/numDepths) '''
    def __checkIfHaveMatrixDim(self, dimname):
        if getattr(self, dimname, None) is None:  # check if attribute exists
            setattr(self, dimname, -1)  # if it doesn't exist
            return False
        return getattr(self, dimname) != -1

    ''' __checkIfDimsMatch: 
    DESCRIPTION: helper to check, if dimensions of all matrices to read-in match
                 initialisation: haveDim = [False False]; oldval = [-1 -1]
                 1st matrix: dimension is stored in oldval, haveDim[dimnum] stays False
                 2nd matrix: if dimensions match, haveDim[dimnum] = True
    INPUT: 
        * mat: newly read-in matrix
        * dimnum: index of dimension (numCompartments, numDepths)
        * haveDim: info from __checkIfHaveMatrixDim as matrix[numDims]
        * oldVal: known dimensions as matrix[numDims] (or -1 if not known)
    OUTPUT: bool '''
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
        haveDim = [self.__checkIfHaveMatrixDim(x) for x in self.DIMS]  # flag, which dims are known
        if all(haveDim): return
        # get dims
        oldVal = (-1) * np.ones([len(self.DIMS)])  # flag, that dim not known for all dims
        # go over all known matrices
        for matname in self.__getAllValuenames(nameClass.matrix):
            readname = self.getVarInfo(matname, nameClass.matrix, nameClass.readname)
            nVar = self.getVarInfo(matname, nameClass.matrix, nameClass.nVar)
            mat = readMatrixFromText(self.filetext, readname, nVar)
            # check for each dim, if they match previous dimensions
            for i in range(0, len(self.DIMS)):
                haveDim, oldVal = self.__checkIfDimsMatch(mat, i, haveDim, oldVal)
            if all(haveDim): return
        raise Exception('\
            \nNumber of Depth levels not defined. \
            \nNeeded as direct input ("number of depth levels = ...") \
            \nor can be derived from resting state conditions.\n')  # numCompartments known from definition

    ''' __init_matrices: initialize all matrices needed '''
    def __init_matrices(self):
        for matname in self.__getAllValuenames(nameClass.matrix):
            if self.getVarInfo(matname, nameClass.matrix, nameClass.nVar) > -1:
                initVal = self.__getInitVal(matname)
                setattr(self, matname, np.ones([self.numCompartments, self.numDepths]) * initVal)
    
    ''' __isInit: return False, if a matrix/value is not in initial conditions any more
                  return True, if is in initial conditions or hasn't been initialized yet '''
    def __isInitMatrix(self, varname, vartype=nameClass.matrix, bold=False):
        var = self.getVarValue(varname, bold)
        if var is None: return True
        initVal = self.__getInitVal(varname, vartype, bold)
        if vartype == nameClass.single: return var == initVal
        else: return (var==initVal).all()
    
    ''' __isInitMatrixValue: return False, if an entry in a matrix is not in initial conditions any more '''
    def __isInitMatrixValue(self, varname, k, d, bold=False):
        mat = self.getVarValue(varname, bold)
        return mat[k,d] == self.__getInitVal(varname)

    ''' __parse: read a single value or matrix (vartype = 'single'/'matrix') from the parameter file '''
    def __parse(self, varname, vartype=nameClass.matrix, bold=False):
        readname = self.getVarInfo(varname, vartype, nameClass.readname, bold)
        wouldThrow = self.getVarInfo(varname, vartype, nameClass.throwException, bold)
        # read in single variable
        if vartype == nameClass.single: 
            typestring = self.getVarInfo(varname, vartype, nameClass.type, bold)
            res = readValFromText(self.filetext, readname, typestring)
            if res is None:
                res = self.getVarInfo(varname, vartype, nameClass.defaultVal, bold)
                if not wouldThrow and not bold: warn(f'{readname} not given. Using defaultVal {varname} = {res}')
        # read in matrix
        elif vartype == nameClass.matrix:
            nVar = self.getVarInfo(varname, vartype, nameClass.nVar, bold)
            res = getattr(self, varname, None)  # matrix with default values, more efficient when given as input
            res = readMatrixFromText(self.filetext, readname, self.numCompartments, self.numDepths, nVar, res)
        # set as variable in self
        self.__setVar(varname, res, bold)
        # add to error message, if unsuccesful
        if wouldThrow and self.__isInitMatrix(varname, vartype, bold): self.__addNewException(readname)
        return res

    ''' __BOLDfail: define what to do if a bold-param is missing '''
    def __BOLDfail(self, varname, vartype):
        readname = self.getVarInfo(varname, vartype, nameClass.readname, bold=True)
        warn(f'{readname} not given. Won\'t calculate BOLD-contrast.')
        self.haveBOLDparams = False
    
    ''' __BOLDcheck: check, that all BOLD-parameters are there '''
    def __BOLDcheck(self, vartype):
        # for all bold-params
        for varname in self.__getAllValuenames(vartype, bold=True):
            if self.__isInitMatrix(varname, vartype, bold=True):
                # E0 can be given as 1) matrix or 2) boldparam -> if 2), init conditions are tolerated
                if varname == 'E0':
                    if hasattr(self.boldparams, 'E0'): continue
                # for every other parameter, init conditions mean read in failed
                self.__BOLDfail(varname, vartype)
                    
    ''' __parse_parameterFile: read all required parameters from the parameter file '''
    def __parse_parameterFile(self, filename):
        self._exception = ''  # throw exception if values are missing (collect all misses before throwing)
        self.filetext = getFileText(filename)  # get total file as string
        # get singular values
        for valuename in self.__getAllValuenames(nameClass.single): self.__parse(valuename, nameClass.single)
        # get balloon matrices
        self.__getBasicMatrixDims()
        self.__init_matrices()  # need init, so that they have correct initVal
        for matname in self.__getAllValuenames(nameClass.matrix): self.__parse(matname)
        # check init conditions (need 2 out of 3 -> can't just use __addNewException) 
        if sum(not self.__isInitMatrix(x) for x in ['V0', 'F0', 'tau0']) < 2: 
            self._exception = self._exception + \
                '\nResting conditions required. Define two columns of F0, V0 or tau0 (see comments for specific format). '
        # get BOLD-parameters
        self.boldparams = { 'gamma0': 2*np.pi*42.58*pow(10,6) }
        # bold matrice
        boldparams_tmp, _, titles = readMatrixFromText(
            self.filetext, 'BOLD', self.numCompartments, numDepths=-1, nVar=0, needCaptions=True)
        if boldparams_tmp is None: warn(f'No BOLD parameters given. Won\'t calculate BOLD-contrast.')
        elif titles is None: warn(f'Names of BOLD-parameters not clear. Won\'t calculate BOLD-contrast.')
        else: 
            for i in range(0, len(titles)): self.boldparams[titles[i]] = boldparams_tmp[:,i]
        # bold single vars
        for valuename in self.__getAllValuenames(nameClass.single, bold=True): 
            self.__parse(valuename, nameClass.single, bold=True)
        # check, if BOLD-params are complete (and warn if not, but don't throw)
        for vartype in nameClass.vartypes[0:1]: self.__BOLDcheck(vartype)
        # throw exception if anything is missing
        if len(self._exception) > 0: raise Exception(self._exception + '\n\n')

# ---------------------------------  CHECK IF F0,V0,tau0 INPUT IS VIABLE  -------------------------------- 
    ''' __countChangedVals: count, how many inserted values are on one layer '''
    def __countChangedVals(self, matrix, varname):
        return sum(matrix != self.__getInitVal(varname) * np.ones(np.size(matrix)))

    ''' __throwIfFailed: raise exception, if FVt input is not viable '''
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
        under = 0  # underdetermined layers
        over = 0  # overdetermined layers
        haveF0 = np.zeros([self.numDepths, 2])  # col1: haveF0; col2: compartment with 2 values (if not haveF0)
        valsPerLayer = self.numCompartments  # 3
        # go through all layers
        for d in range(0, self.numDepths):
            # get number of elements
            numF = self.__countChangedVals(self.F0[:,d], 'F0')
            numV = self.__countChangedVals(self.V0[self.VENULE:self.numCompartments,d], 'V0')
            numT = self.__countChangedVals(self.tau0[self.VENULE:self.numCompartments,d], 'tau0')
            # check if error (need exactly 3 vals per layer, max one of them can be F)
            if numF + numV + numT != valsPerLayer or numF > 1:
                under += numF + numV + numT < valsPerLayer
                over += numF + numV + numT > valsPerLayer or numF > 1
                continue
            # save if have F0
            haveF0[d,0] = numF == 1
            # check that at least one value for venule, vein compartments each
            for k in range(self.VENULE, self.numCompartments):
                change = \
                      int(not self.__isInitMatrixValue('V0', k, d)) \
                    + int(not self.__isInitMatrixValue('tau0', k, d)) 
                under += int(change==0)
                # save compartment where F0 can be calculated
                if not haveF0[d,0] and change == 2:
                    haveF0[d,1] = int(k)
        # throw exception if any layer is faulty
        self.__throwIfFailed(under, over)
        return haveF0

# ----------------------------------  FILL V0, F0, TAU0  ---------------------------------------
    def __calcV(self, k,d): self.V0[k,d] = self.F0[k,d] * self.tau0[k,d]
    def __calcF(self, k,d): self.F0[k,d] = self.V0[k,d] / self.tau0[k,d]
    def __calcTau(self, k,d): self.tau0[k,d] = self.V0[k,d] / self.F0[k,d]

    ''' __getThirdValueVFt: get one value for {tau0 = V0/F0} 
                            hardVal: gives possibility to hardcode, which value to calculate 
                            otherwise: finds out automatically, which value to calculate '''
    def __getThirdValueVFt(self, k,d, hardVal=None):
        if hardVal is None:
            f = self.__isInitMatrixValue('F0', k, d)
            v = self.__isInitMatrixValue('V0', k, d)
            t = self.__isInitMatrixValue('tau0', k, d)
        else:
            f = hardVal[0] in ['f', 'F']
            v = hardVal[0] in ['v', 'V']
            t = hardVal[0] in ['t', 'T']
        if v and not f and not t: self.__calcV(k,d)
        if f and not v and not t: self.__calcF(k,d)
        if t and not v and not f: self.__calcTau(k,d)
        
    ''' __fillVFt: get all values for V0, F0, tau0 (assume requirements are met) '''
    def __fillVFt(self, haveF0):
        # go layer-wise, start in lowest layer because of flow-dependencies
        for d in range(self.numDepths-1, -1, -1):
            # make sure there is exactly one column of F0
            if not haveF0[d,0]:
                k = int(haveF0[d,1])  # the commpartment, where F can be calculated
                self.__getThirdValueVFt(k,d)
            # fill F0 in all compartments
            self.__makeFlowMeetConditions(d)
            # get tau0, V0 by tau0 = V0/F0
            for k in range(0, self.numCompartments):
                if k==self.ARTERIOLE: continue
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
    
    ''' __findChangedVal: return first compartment of list where flow was changed since init of matrix '''
    def __findChangedVal(self, k_list, d):
        for k in k_list:
            if not self.__isInitMatrixValue('F0', k, d): 
                return k
    
    ''' __makeFlowMeetCondition_oneCompartment: make flow in one compartment meet resting state conditions '''
    def __makeFlowMeetCondition_oneCompartment(self, k, k_withFlow, d):
        if not self.__needDeeperLayerFormula([k, k_withFlow], d):  # if higher layers of VEIN aren't involved
            self.F0[k,d] = self.F0[k_withFlow,d]  # 
        elif k_withFlow == self.VEIN:  # if need deeper layer because k_withFlow is VEIN
            self.F0[k,d] = self.F0[self.VEIN,d] - self.F0[self.VEIN,d+1]  # arterial=venule flow gives difference between layers
        else:  # if need deeper layer because k is VEIN
            self.F0[k,d] = self.F0[k_withFlow,d] + self.F0[self.VEIN,d+1]  
    
    ''' __makeFlowMeetConditions: make flow in all compartments meet resting state conditions 
                                  assume, that exactly one compartment already has flow 
                                  -> get flow for the other compartments '''
    def __makeFlowMeetConditions(self, d):
        k_list = [self.ARTERIOLE, self.VENULE, self.VEIN]
        k_withFlow = self.__findChangedVal(k_list, d)  # the compartment, where flow was defined first
        for k in range(0, self.numCompartments):
            if k==k_withFlow: continue
            self.__makeFlowMeetCondition_oneCompartment(k, k_withFlow, d)

# -----------------------------  MAKE SURE Q CAN BE CALCULATED  -------------------------------
    ''' __E0_is_KxD_matrix: make sure, E0 was given as matrix and not as part of boldparams '''
    def __E0_is_KxD_matrix(self):
        if self.__isInitMatrix('E0'): return False
        if (self.E0>0.7).any() or (self.E0<0).any(): return False
        # make sure, not any other matrix was accidentally copied to E0
        for attr in self.boldparams:
            if attr == 'E0': continue
            for d in range(0, self.numDepths):
                if (self.E0[:,d]==self.boldparams[attr]).all(): return False
        # if all checks check out, assume E0 was read-in properly
        return True
    
    ''' __checkQ: make sure, given input is sufficient to calculate ox-extraction (q, dq) '''
    def __checkQ(self):
        if not self.__isInitMatrix('n'): self.oxmode = self._OX_n
        elif self.__E0_is_KxD_matrix(): self.oxmode = self._OX_E0
        else: 
            self.oxmode = self._OX_m
            self._exception = (f"\ndHb-content can not be calculated with given input. "
                               f"\nGive either "
                               f"{self.getVarInfo('n', nameClass.matrix, nameClass.readname)} or "
                               f"{self.getVarInfo('E0', nameClass.matrix, nameClass.readname)} "
                               f"as matrix [{self.DIMS[0]}, {self.DIMS[1]}].\n"
                               f"Alternatively, give CMRO2 as inputfile into 'Input_Timeline'.\n\n")

# -----------------------------------  CHANGE-FUNCTIONS  ----------------------------------------
    ''' __changeSingleValOfMatrix: change single entry in a matrix <varname> '''
    def __changeSingleValOfMatrix(self, varname, new_val, index, bold=False):
        mat = self.getVarValue(varname, bold)
        mat[index[0], index[1]] = new_val
        self.__setVar(varname, mat, bold)
        return mat
 
    ''' changeVar: change value of a single variable. 
        Calculate other FVT value if FVT changed.
        Change only specific index [numCompartments, numDepths] of matrix, if index given (change entire matrix if index==[]). 
        returns, if change was successful '''
    def changeVar(self, varname, new_val, index=[], dependentFVT='t'):
        vartype, bold = self.findVarname(varname)  # find out where the variable is stored
        if vartype is None or bold is None: 
            warn(f"Tried to change variable of unknown name. {varname} not found in Model_Parameters.")
            return False
        if not self.getVarInfo(varname, vartype, 'changeable', bold):
            warn(f"Tried to change {varname}. Unchangeable variable. Ignoring change. Create new read-in instead.")
            return False
        # if matrix 
        if vartype==nameClass.matrix and index!=[]:
            if varname in ['F0', 'V0', 'tau0']: 
                # V0, tau0 can not be changed in arteriole compartment
                if varname in ['V0', 'tau0'] and index[0] == self.ARTERIOLE:
                    warn(f"Tried to change {varname} in arteriole compartment. Irrelevant parameter. Change in venule/vein compartment or consider changing flow.")
                    return False
                # change values and all dependencies
                self.__changeVFt(varname, new_val, index, dependentFVT)
            else:  # matrix, but not FVt
                self.__changeSingleValOfMatrix(varname, new_val, index, bold)
        else:  # if single variable or entire matrix given 
            self.__setVar(varname, new_val, bold)
        return True
    
    ''' __changeVFt: change a single value of VFt. Do all follow-up changes:
                    1) make flow meet conditions
                    2) change dependentVar for all flow changes '''
    def __changeVFt(self, varname, new_val, index, dependentVar='tau0'):
        dependentVar = self.__checkDependent(varname, dependentVar)
        k = index[0]
        d = index[1]
        # change value and dependent var
        self.__changeSingleValOfMatrix(varname, new_val, index)
        self.__getThirdValueVFt(k, d, hardVal=dependentVar) 
        # apply flow conditions
        if 'f' in [varname[0], dependentVar[0]] or 'F' in [varname[0], dependentVar[0]]:
            # flow is changed first, the other variable is dependent
            if dependentVar[0] in ['f', 'F']:
                dependentVar = varname
                varname = 'F0'
            # get flow in current depth for all compartments
            k_withFlow = k  # has already been changed
            for k in range(0, self.numCompartments):
                if k==k_withFlow: continue
                self.__makeFlowMeetCondition_oneCompartment(k, k_withFlow, d)
                self.__getThirdValueVFt(k, d, hardVal=dependentVar)
            # get flow for VEIN in lower depths
            for d in range(index[1]-1, -1, -1):
                self.__makeFlowMeetCondition_oneCompartment(self.VEIN, self.VENULE, d)
                self.__getThirdValueVFt(self.VEIN, d, hardVal=dependentVar)
    
    ''' __dependentCheckFailed: define what to do if dependency not clear '''
    def __dependentCheckFailed(self, varname, case):
        w = ['Change of F0, V0 or tau0 requires info about which dependent variable to change as well [F0, V0 or tau0]. Using default: ', \
             'Dependent variable of V0/F0 = tau0 is given as the variable to be changed. Using default: ']
        if case==1 and varname == 'tau0': dependent = 'V0'
        else: dependent ='tau0'
        warn(w[case] + dependent)
        return dependent

    ''' __checkDependent: check, if dependentVar is one of [F0, V0, tau0] '''
    def __checkDependent(self, varname, dependentVar):
        if dependentVar[0] in ['f', 'F']: dependentVar = 'F0'
        elif dependentVar[0] in ['v', 'V']: dependentVar = 'V0'
        elif dependentVar[0] in ['t', 'T']: dependentVar = 'tau0'
        else: dependentVar = self.__dependentCheckFailed(varname, case=0)
        if varname == dependentVar: dependentVar = self.__dependentCheckFailed(varname, case=1)
        return dependentVar


