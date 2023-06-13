

import copy
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from warnUsr import warn
from readFile import readFiles

from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters

def repmat3(mat, new_size, dim=2):  # turn 2D into 3D matrix
    if dim==0: return mat.reshape([1, mat.shape[0], mat.shape[1]]).repeat(new_size, dim)  # [K,D] into [x,K,D]
    if dim==1: return mat.reshape([mat.shape[0], 1, mat.shape[1]]).repeat(new_size, dim)  # [K,D] into [K,x,D]
    if dim==2: return mat.reshape([mat.shape[0], mat.shape[1], 1]).repeat(new_size, dim)  # [K,D] into [K,D,x]
def repmat2(mat, new_size, dim=0):  # turn 1D into 2D matrix 
    if dim==0: return mat.reshape([1, mat.shape[0]]).repeat(new_size, dim)  # [D] into [K,D]
    if dim==1: return mat.reshape([mat.shape[0], 1]).repeat(new_size, dim)  # [K] into [K,D]

''' Y_Properties: define the structure of the result vector y
        * y is a 1D array
        * stores inits/ results of <functions> such as 'n_excitation'
        * assumes all <functions> have <numDepths> values -> total size of y = nFunctions * nDepths
        * exception: 'f', 'v', 'q' have 2*nDepths values each (venule and vein; f_arteriole is a separate function)
        * Y_properties.indice[function_name][0] stores first index where result of <function> is stored in y 
    ATTRIBUTES:
        * Y_Properties.funnames : list of all implemented <functions> (['n_excitation', 'n_inhibition', ...])
        * Y_Properties.indice : stores index and initial val of each function for each depth ( indice.BOLD_0 = [index, initVal] )
        * Y_Properties.y0 : vector with initital values
        * Y_Properties.compartmented : list of functions that have 2 compartments (['f', 'v', 'q'])
        * Y_Properties.compartments : list of compartments (['_venule', '_vein'])
        * Y_Properties.flowdependent : list of functions where result depends on the flow direction
    METHODS:
        * getFunname: get name of dict entry for specific function and depth (eg 'n_excitation_0' for d=0) 
        * getYindex: get index in y for specific function and depth 
        * getYinit: get initial value for specific function and depth 
        * getNumCompartments: get info, if <function> has 2 compartments (venule/vein) or just 1
        * getNumEls: get space that <function> occupies in y
        * getVal: needs vector y as input, returns value for specific function and depth 
        * getVec: needs vector y as input, returns entire vector for specific function (all depths) 
                options: time='off' -> assume y is 1D
                         time='on' -> assume y is 2D, return 2D vector (<function> over entire time) '''
class Y_Properties:
    def __init__(self, params: Model_Parameters):
        self.params = params
        self.__defineYproperties()

    def getFunname(self, funname, k=0, d=0): return f'{funname}{self.getCompartmentName(funname, k)}_{d}'
    def getCompartmentName(self, funname, k):  # k in [0,1] for all functions
        if funname in self.compartmented: return self.compartments[k]
        else: return '' 
    def __getIndexVal(self, funname, k, d):
        if self.getFunname(funname, k=k, d=d) in self.indice:
            return self.indice[self.getFunname(funname, k=k, d=d)]
        return [-1, -1]
    def getYindex(self, funname, k=0, d=0): return self.__getIndexVal(funname, k, d)[0]
    def getYinit(self, funname, k=0, d=0): return self.__getIndexVal(funname, k, d)[1]
    def getNumCompartments(self, funname):
        if funname in self.compartmented: return len(self.compartments)
        else: return 1
    def getNumEls(self, funname): return self.params.numDepths * self.getNumCompartments(funname)
    def getVal(self, y, funname, k=0, d=0): return y[self.getYindex(funname, k, d)]
    def getVec(self, y, funname, time='off'):
        i0 = self.getYindex(funname)
        if time=='off': return y[i0:i0+self.getNumEls(funname)]
        return y[i0:i0+self.getNumEls(funname), :]
    def getCompartment(self, y, funname, k): return self.getVec(y, funname)[k*self.params.numDepths:(k+1)*self.params.numDepths]

    def __defineYproperties(self):
        self.funnames = ['n_excitation', 'n_inhibition', 'vaso_active', 'f_arteriole', 'f', 'v', 'q', 'BOLD']
        self.compartmented = ['f', 'v', 'q']
        self.compartments = ['_venule', '_vein']
        self.flowdependent = ['f']
        self.indice = dict()
        self.y0 = np.zeros( (len(self.funnames) + len(self.compartmented)*(len(self.compartments)-1) ) * self.params.numDepths)
        index = 0
        for funname in self.funnames:
            if funname in self.compartmented: K = len(self.compartments)
            else: K = 1
            for k in range(0, K):
                for d in range(0, self.params.numDepths):
                    self.__defineInitValue(funname, k, d, index)
                    index = index + 1

    def __setInitValue(self, funname, k, d, index, y0): 
        if index==-1: return y0
        self.y0[index] = y0
        self.indice[self.getFunname(funname, k, d)] = [index, self.y0[index]]  # self.BOLD_d = [index, initVal]  etc
        return y0
    def __defineInitValue(self, funname, k, d, index):
        # trivial values
        if funname in ['n_excitation', 'n_inhibition', 'vaso_active', 'BOLD']: y0 = 0
        if funname in ['f_arteriole', 'q']: y0 = 1
        if 'y0' in locals(): return self.__setInitValue(funname, k, d, index, y0)
        # check if non-trivial value has already been calculated
        y1 = self.getYinit(funname, k=k, d=d)
        if y1 > -1: return y1
        # set preliminaries for non-trivial values
        p, D, F0, alpha = self.params, self.params.numDepths, self.params.F0, self.params.alpha
        kk = k+p.VENULE  # different indexing for X0 matrice
        # calculate non-trivial values
        if funname == 'f' and kk==p.VENULE: y0 = F0[p.ARTERIOLE,d]/F0[kk,d]
        if funname == 'f' and kk==p.VEIN: 
            if d == D-1: y0 = F0[p.ARTERIOLE,d]/F0[kk,d]
            else: y0 = F0[p.ARTERIOLE,d]/F0[kk,d] + F0[kk,d+1]/F0[kk,d] * self.__defineInitValue(funname, k, d+1, -1)
        if funname == 'v': y0 = self.__defineInitValue('f', k, d, -1) ** alpha[k, d]
        if 'y0' in locals(): return self.__setInitValue(funname, k, d, index, y0)
        # warn, if value not defined
        warn(f'initial Value for {funname} not implemented. Returning -1.')
        return -1


''' hemodynamic_model_vector: class for the vector y that is processed during the IVP
        ATTRIBUTES: for each <funname>: Y_Vector.funname = partial vector 
        FUNCTIONS: (runs on entire fun, all compartments at once) 
            * previousCompartment(): return vector of flow that lies in previous compartments (arteriole, venule)
            * deeperLayer(funname): return vector of <funname> transformed into deeper layer (deepest layer=0, all other layers are rotated 1 up) '''
class Y_Vector:
    def __init__(self, y, y_properties: Y_Properties):
        self.__y_input, self.__y_props = y, y_properties
        for funname in y_properties.funnames: setattr(self, funname, y_properties.getVec(y, funname))

    def deeperLayer(self, funname, noZero=False):
        if noZero: default = 1
        else: default = 0
        vec = np.append(getattr(self, funname)[1:], default)  # rotate everything 1 up, append 0
        for i in range(0, self.__y_props.getNumCompartments(funname)-1): vec[(i+1)*self.__y_props.params.numDepths-1] = default  # set all other deepest layers to 0
        return vec
    def previousCompartment(self):
        funname, k_venule = 'f', 0
        vec_venule = self.__y_props.getCompartment(self.__y_input, funname, k_venule)
        vec_arteriole = getattr(self, 'f_arteriole')
        return np.append(vec_arteriole, vec_venule)
    def getCompartment(self, funname, k): return self.__y_props.getCompartment(self.__y_input, funname, k)


''' hemodynamic_model_constants: store a few constants for hemodynamic model calculations (to only calculate them once) 
    ATTRIBUTES:
        * hemodynamic_model_constants.funnames : stores names of functions for which the constants are needed
                funnames.fvq : flow, volume and q
                funnames.BOLD: BOLD
                * separation is useful because different parts are needed 
        * hemodynamic_model_constants.parts : defines, which parts of the function need constants (eg 'previous_compartment', 'deeper_layer' etc)
                * also separated into parts.fvq and parts.BOLD
        * hemodynamic_model_constants.consts : the actual constants in the form of: consts.f.deeper_layer = mat[K,D,nDir]  (K, nDir=1 if not relevant) '''
class hemodynamic_model_constants:
    def __init__(self, \
                 params: Model_Parameters, \
                 y_properties: Y_Properties ):
        self.params, self.__y_prop = params, y_properties
        self.__calculateConstants()
    
    def __setConst(self, funname, part, const): 
        if not funname in self.__y_prop.flowdependent: const = const.reshape(const.size)  # turn into 1D matrix
        self.consts[funname][part] = const
        return const
    def getConst(self, funname, part, k=0, d=0, dir=0): return self.consts[funname][part][k,d,dir]  # k in [numComp for that function]

    def __calculateConstants(self):
        # set preliminaries
        p, b, K, D, nDir = self.params, self.params.boldparams, self.params.numCompartments, self.params.numDepths, len(self.params.FLOWDIRS) 
        self.funnames, self.parts, self.consts = dict(), dict(), dict()
        # define structure of required constants
        self.funnames['fvq'] = ['f', 'v', 'q']
        self.funnames['BOLD'] = ['BOLD']
        # define indice of parts
        self.parts['fvq'] = ['same', 'prevComp', 'deeperLayer']  # prevComp: previous compartment
        self.parts['BOLD'] = ['q', 'qv', 'v']
        # partial constants that are used several times
        F_denom = p.vet[p.VENULE:K,:,:] * repmat3(p.F0[p.VENULE:K,:],nDir) + repmat3(p.V0[p.VENULE:K,:],nDir)
        V0_ = p.V0 * D / 100  # transformation from total volume to blood volume fraction
        sV0 = np.sum(V0_, 0)
        seV = np.sum(V0_ * repmat2(b['epsilon'], D, dim=1), 0)
        H0 = 1 / (1 - sV0 + seV)
        # calculate constants
        for funtype, funlist in self.funnames.items():  # for ['fvq', 'BOLD']
            for funname in funlist:
                self.consts[funname] = dict()
                for part in self.parts[funtype]:
                    self.__defineConst(funname, part, F_denom, H0, sV0)

    def __defineConst(self, funname, part, F_denom_3x, H0, sV0):
        # check if has been calculated
        if part in self.consts[funname]: return self.consts[funname][part]
        if funname == 'q': return self.__setConst(funname, part, self.__defineConst('v', part, F_denom_3x, H0, sV0))  # constants for v and q are equal
        # notations 
        p, pb, D, K, X, nDir = self.params, self.params.boldparams, self.params.numDepths, self.params.numCompartments, len(self.__y_prop.compartments), len(self.params.FLOWDIRS)
        if funname not in self.__y_prop.flowdependent: nDir = 1  # only flow depends on direction
        # compartment notation: use k if [arteriole, venule, vein], use x if [venule, vein]
        k0 = p.VENULE  # first compartment for k -> x conversion
        x_venule, x_vein = p.VENULE-k0, p.VEIN-k0  # compartment indice for x-indexed matrice (use p.COMPARTMENT otherwise)
        F0_3x, V0_3x, tau0_x, vet_3x = repmat3(p.F0[k0:K,:], nDir), repmat3(p.V0[k0:K,:], nDir), p.tau0[k0:K,:], p.vet[k0:K,:,:]  # 3D with [X,D,nDir]
        # calculate value
        if funname == 'f':
            c = np.zeros([X,D,nDir])  # need init for deeperLayer
            if part=='same': c = V0_3x / F_denom_3x
            if part=='prevComp': 
                F0_VENULE = p.F0[p.VENULE,:].reshape([1,D,1]).repeat(X, 0).repeat(nDir, 2)
                c = vet_3x * F0_VENULE / F_denom_3x
            if part=='deeperLayer': c[x_vein, 0:D-1, :] = vet_3x[x_vein, 0:D-1, :] * F0_3x[x_vein, 1:D, :] / F_denom_3x[x_vein, 0:D-1, :]
        if funname == 'v':
            c = np.zeros([X,D])  # need init for deeperLayer
            if part=='same': c = - 1 / tau0_x
            if part=='prevComp': c = p.F0[p.ARTERIOLE:p.VENULE+1,:] / p.F0[p.VENULE:p.VEIN+1,:] / tau0_x
            if part=='deeperLayer': c[x_vein, 0:D-1] = p.F0[p.VEIN, 1:D] / p.F0[p.VEIN, 0:D-1] / tau0_x[x_vein, 0:D-1]
        # BOLD
        if funname == 'BOLD':
            if part=='q':  # c1,2,3 ... [K] ; H ... [D]
                c1 = 4.3 * pb['dXi'] * pb['Hct'] * pb['gamma0'] * pb['B0'] * pb['E0'] * pb['TE']
                c = repmat2(H0, X, dim=0) * repmat2(c1[k0:K], D, dim=1) * p.V0[k0:K,:] * D / 100 * (1 - repmat2(sV0, X, dim=0) )
            if part=='qv':
                c2 = pb['epsilon'] * pb['r0'] * pb['E0'] * pb['TE']
                c = repmat2(H0, X, dim=0) * repmat2(c2[k0:K], D, dim=1) * p.V0[k0:K,:] * D / 100
            if part=='v':
                c3 = 1 - pb['epsilon']
                c = repmat2(H0, X, dim=0) * repmat2(c3[k0:K], D, dim=1) * p.V0[k0:K,:] * D / 100
        return self.__setConst(funname, part, c)


''' hemodynamic_model_functions: class for functions to solve the IVP
        FUNCTIONS:
            * hemodynamic_model(t, y): main function, calculates the next time step of the IVP for all stages (neural, balloon, BOLD)
                - INPUT: 1D y (as defined by Y_Properties)
                - OUTPUT: 1D y_new (same structure as y) 
            * setStimulus(S): set the stimulus as attribute (serves as constraint for the IVP)
            * setModelType(modelType): decide between options:
                - 'contaminated' (by deeper layers)
                - 'clean' (no contamination -> VASO signal appearance)
            * setCalculationMode(calculationMode): decide between options:
                - 'immediate' -> use results from previous stages to compute values of following stages in same time step (eg use flow[t+1] to calculate volume[t+1])
                - 'last' -> use entire y to calculate y_new (without using intermediate results from the same time step) '''
class hemodynamic_model_functions:
    def __init__(self, \
                 nparams: Neural_Parameters, \
                 params: Model_Parameters, \
                 y_properties: Y_Properties, \
                 consts: hemodynamic_model_constants ):
        self.params, self.nparams, self.__y_prop, self.consts = params, nparams, y_properties, consts
        self.init()

    def init(self, modelType='contaminated', calculationMode='immediate'):
        self.setModelType(modelType)  # default with contamination by deeper layers
        self.setCalculationMode(calculationMode)  # use results from same time step for value computation of following stages in same time step
        self.__t, self.__v = 0, 0  # save values for next step
    def setStimulus(self, S): self.S = S  # should be [D,nT]
    def setModelType(self, modelType):  self.modelType = modelType  # model types: 'contaminated', 'clean'  -> contaminated by deeper layers or not
    def setCalculationMode(self, calculationMode): self.calculationMode = calculationMode  
    # modes: 'immediate', 'last'  -> use already calculated values of same time step or stay clean, process entire vector
    
    def __getFlowDir(self, Y:Y_Vector, dt): return ((Y.v >= self.__v) * dt > 0) + ((Y.v < self.__v) * dt < 0)  # True if inflation
    def __getCurrentC(self, isInflation):  # takes __getFlowDir as input, returns constants-object for current flowdir
        C, p = copy.deepcopy(self.consts.consts), self.params
        for flowFun in self.__y_prop.flowdependent:  # for all functions that depend on flowdir
            for part in self.consts.parts['fvq']:
                isInflation = isInflation.reshape([C[flowFun][part].shape[0], C[flowFun][part].shape[1]])  # isInflation = [2*D]
                C[flowFun][part] = np.int32(isInflation) * C[flowFun][part][:,:,p.INFLATION] + np.int32(~isInflation) * C[flowFun][part][:,:,p.DEFLATION]
                C[flowFun][part] = C[flowFun][part].reshape(C[flowFun][part].size)  # get original size
        return C  # return entire construct
    def __updateVec(self, funname, new_vec):
        i0 = self.__y_prop.getYindex(funname)
        self.__y_new[i0:i0+self.__y_prop.getNumEls(funname)] = new_vec

    def __executeFun(self, t, Y:Y_Vector, C, funname, dt):
        p, n, K, D, x_venule = self.params, self.nparams, self.params.numCompartments, self.params.numDepths, 0 
        isContaminated = np.int32(self.modelType == 'contaminated')  # defines, if we need contamination by deeper layers
        if funname == 'n_excitation': dx = n.sigma * Y.n_excitation - n.mu * Y.n_inhibition + n.C * self.S[:,np.int32(np.round(t/p.dt))]
        if funname == 'n_inhibition': dx = n.lambd * (Y.n_excitation - Y.n_inhibition)
        if funname == 'vaso_active': dx = Y.n_excitation - n.c1 * Y.vaso_active
        if funname == 'f_arteriole': dx = n.c2 * Y.vaso_active - n.c3 * (Y.f_arteriole - 1)
        if funname == 'f': return \
            C[funname]['same'] * np.power( Y.v, 1/p.alpha[p.VENULE:K,:].reshape(Y.v.shape[0]) ) \
            + C[funname]['prevComp'] * Y.previousCompartment() \
            + C[funname]['deeperLayer'] * Y.deeperLayer('f') * isContaminated
        if funname == 'v': dx = \
            C[funname]['same'] * Y.f \
            + C[funname]['prevComp'] * Y.previousCompartment() \
            + C[funname]['deeperLayer'] * Y.deeperLayer('f') * isContaminated
        if funname == 'q': 
            nn = p.n[p.VENULE,:].reshape(Y.f_arteriole.shape[0])
            prevComp_q_venule = 1/nn + 1/Y.f_arteriole - 1/nn/Y.f_arteriole
            prevComp_q_vein = Y.getCompartment('q', x_venule) / Y.getCompartment('v', x_venule)
            prevComp_q = np.append(prevComp_q_venule, prevComp_q_vein)
            dx = \
                C[funname]['same'] * Y.f * Y.q / Y.v \
                + C[funname]['prevComp'] * Y.previousCompartment() * prevComp_q \
                + C[funname]['deeperLayer'] * Y.deeperLayer('f') * Y.deeperLayer('q') / Y.deeperLayer('v',True) * isContaminated
        if funname == 'BOLD': 
            vec = C[funname]['q'] * (1-Y.q) + C[funname]['qv'] * (1-Y.q/Y.v) + C[funname]['v'] * (1-Y.v)
            return vec[0:D] + vec[D:2*D]  # sum over compartments
        if funname in ['n_excitation', 'n_inhibition', 'vaso_active']: conversionType = 'Euler'
        if funname in ['f_arteriole', 'v', 'q']: conversionType = 'log-normal'
        return self.__nextValFromD(getattr(Y,funname), dx, dt, conversionType)
    
    def __nextValFromD(self, x_old, dx, dt, conversionType='Euler'):
        if conversionType == 'Euler': return x_old + dx * dt
        if conversionType == 'log-normal': return x_old * np.exp(dx * dt / x_old)

    def hemodynamic_model(self, t, y):
        Y = Y_Vector(y, self.__y_prop)  # turn y into a more readable form
        v_last, dt = copy.deepcopy(Y.v), t - self.__t
        C = self.__getCurrentC(self.__getFlowDir(Y, dt))
        self.__y_new = y  # overwrite step by step
        for funname in self.__y_prop.funnames: 
            if self.calculationMode=='immediate': Y = Y_Vector(self.__y_new, self.__y_prop)
            self.__updateVec(funname, self.__executeFun(t,Y,C,funname,dt))  # get next value into y_new
        self.__t, self.__v = t, copy.deepcopy(v_last)
        return self.__y_new
    

''' hemodynamic_model_forward: class to compute the hemodynamic model as initial value problem (IVP) with a stimulus S(t) [D,nT] as input
        FUNCTIONS: 
            * solveIVP(S): calculates the entire model
                INPUT: stimulus S[nT] or S[D,nT]
                OUTPUT: 
                    - t[nT+1]
                    - y[x,nT+1] ; numels and meaning of elements on first dim is defined by y_properties
                        # to get result of specific function, use self.y_properties.getVec(y, <funname>, 'on')
                    - BOLD-signal[D,nT+1]
            * plotFun(funname, y): plot a specific function from result vector y '''
class hemodynamic_model_forward:
    def __init__(self, \
                 params: Model_Parameters, \
                 neural_params: Neural_Parameters ):
        self.params, self.neural_params = params, neural_params
        self.y_properties = Y_Properties(self.params)
        self.consts = hemodynamic_model_constants(self.params, self.y_properties)
        self.functions = hemodynamic_model_functions(self.neural_params, self.params, self.y_properties, self.consts)
    
    def __checkY0(self, y0):
        if y0 is None: return self.y_properties.y0
        if y0.ndim != 1 or y0.shape[0] != len(self.y_properties.y0): 
            warn(f'SolveIVP: y0 should satisfy Y_conditions (shape = [{len(self.y_properties.y0)}])')
            return None
        return y0

    def solveIVP(self, stimulus, y0=None, modelType='contaminated'):
        y0 = self.__checkY0(y0)
        self.functions.init(modelType=modelType)
        if stimulus.ndim == 1: stimulus = repmat2(stimulus, self.params.numDepths, dim=0)  # [D,nT]
        self.functions.setStimulus(stimulus)
        nT = stimulus.shape[1]
        y = np.zeros([len(self.y_properties.y0), nT+1])
        y[:,0] = y0
        for t in range(1, nT+1): 
            y[:,t] = self.functions.hemodynamic_model((t-1)*self.params.dt, copy.deepcopy(y[:,t-1]))
        return np.linspace(0, nT+1, nT+1), y, self.y_properties.getVec(y, 'BOLD', 'on')  # t, y, signal
    
    def plotFun(self, y, funname):
        data = self.y_properties.getVec(y, funname, 'on')
        _, axs = plt.subplots(self.params.numDepths)
        for d in range(0, self.params.numDepths):
            axs[d].plot(data[d,:], label='venule')
            if funname in self.y_properties.compartmented: axs[d].plot(data[d+self.params.numDepths,:], label='vein')
            axs[d].grid(True)
        axs[0].set_title(funname)
        if funname in self.y_properties.compartmented: axs[0].legend()


''' =============================  EXECUTE ======================== '''
if __name__ == '__main__':
    # parameter files
    parameter_file = "/Havliceks_Implementation2019_example1.txt"
    neural_parameter_file = "/NeuralParameters_210812.txt"
    input_function_file = parameter_file
    params, neural_params, input_TL = readFiles(parameter_file, neural_parameter_file, input_function_file)
    # calculate model
    IVP = hemodynamic_model_forward(params, neural_params)
    t, y, signal = IVP.solveIVP(input_TL.stimulus)
    # plot specific funs
    plotfuns = ['n_excitation', 'f_arteriole', 'f', 'v', 'BOLD']
    for funname in plotfuns: IVP.plotFun(y, funname)
    plt.show()
    print('end')
