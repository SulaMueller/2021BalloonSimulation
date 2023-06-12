
''' work in progress '''

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import pylops
from pylops.utils import dottest
from pylops.optimization.basic import lsqr

import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from class_NeuralParameters import Neural_Parameters
from class_ModelParameters import Model_Parameters, makeDict
from class_inputTimeline import Input_Timeline
from class_Plots import Plots


#print('Convolution operator', Cop)
#dottest(Cop, verb=True)  # check, that forward and adjoint operators are correct

class Ops_hemodynamicBOLDmodel:
    def __init__(self, \
                nparams: Neural_Parameters, \
                params: Model_Parameters
                ):
        self.nparams = nparams
        self.params = params
        self.__opsFlag = [-1,-1,-1]  # flag, if operators are defined [K,D,T]
        self.__setKDT(setOps=False, init=True)
    
    def __setKDT(self, K=-1, D=-1, T=-1, setConsts=True, setOps=True, init=False):
        # check cases
        k = K>0 and K != self.params.numCompartments
        d = D>0 and D != self.params.numDepths
        t = T>0 and T != self.params.nT
        # set parameters
        if k: self.params.numCompartments = K
        if d: self.params.numDepths = D
        if t: self.params.nT = T
        # run following operations
        if (k or d) and setConsts or init: self.__defineConstants()
        if (k or d or t) and setOps: self.__defineOperators()

    # =================================== DEFINE CONSTANTS ================================
    def __define_BOLDconstants(self):
        p, b = self.params, self.params.boldparams
        K, D = p.numCompartments, p.numDepths
        c, H0, sV0 = np.zeros([3, K]), np.zeros([D]), np.zeros([D])
        for k in range(0, K):
            if 'E0' in b: e0 = b['E0'][k]
            else: e0 = p.E0[k,:]
            c[0,k] = 4.3 * b['dXi'] * b['Hct'][k] * b['gamma0'] * b['B0'] * e0 * b['TE']
            c[1,k] = b['epsilon'][k] * b['r0'][k] * e0 * b['TE']
            c[2,k] = 1 - b['epsilon'][k]
        # H0, sum of V0
        for d in range(0, D):
            sev = 0  # sum of epsilon * V0
            for k in range(0, K):
                v = p.V0[k,d] * D / 100  # transformation from total volume to blood volume fraction
                sV0[d] += v  # has been initialized as 0
                sev += v * b['epsilon'][k]
            H0[d] = 1 / (1 - sV0[d] + sev)
        return H0, c, sV0

    def __define_flowscaling(self):
        p = self.params
        K, D, numDirs = p.numCompartments, p.numDepths, 2
        flowscaling = np.zeros([K, D, numDirs])
        for k in range(p.VENULE, K):
            for d in range(0, D):
                for flowdir in range(0,numDirs):
                    flowscaling[k,d,flowdir] = p.F0[k,d] * p.vet[k,d,flowdir] + p.V0[k,d]
        return flowscaling
        
    def __defineConstants(self):
        p = self.params
        K, D, numDirs = p.numCompartments, p.numDepths, 2
        # init constants
        listOfConsts_kd = ['A', 'B', 'C', 'K', 'L', 'M']
        listOfConsts_kddir = ['D', 'E', 'G']
        self.consts = {}
        for key in listOfConsts_kd:
            self.consts[key] = np.zeros([K, D])
        for key in listOfConsts_kddir:
            self.consts[key] = np.zeros([K, D, numDirs])
        # get constants for all k,d
        H0, c, sV0 = self.__define_BOLDconstants()
        fs = self.__define_flowscaling()
        for k in range(0, K):
            for d in range(0, D):
                noDeep = k<K-1 or d==D-1
                # BOLD constants
                self.consts['A'][k,d] = H0[d] * c[0,k] * p.V0[k,d] * D / 100 * (1 - sV0[d])
                self.consts['B'][k,d] = H0[d] * c[1,k] * p.V0[k,d] * D / 100
                self.consts['C'][k,d] = H0[d] * c[2,k] * p.V0[k,d] * D / 100
                # Balloon constants
                if k==0: continue
                for dir in range(0, numDirs):
                    self.consts['D'][k,d,dir] = p.V0[k,d] / fs[k,d,dir]
                    self.consts['E'][k,d,dir] = p.vet[k,d,dir] * p.F0[1,d] / fs[k,d,dir]
                    if noDeep: self.consts['G'][k,d,dir] = 0
                    else: self.consts['G'][k,d,dir] = p.vet[k,d,dir] * p.F0[2,d+1] / fs[k,d,dir]
                self.consts['K'][k,d] = -1/p.tau0[k,d]
                self.consts['L'][k,d] = p.F0[k-1,d] / p.F0[k,d] / p.tau0[k,d]
                if noDeep: self.consts['M'][k,d] = 0
                else: self.consts['M'][k,d] = p.F0[k,d+1] / p.F0[k,d] / p.tau0[k,d]
    
    # =================================== DEFINE OPERATORS ================================
    ''' operator and matrix definitions: 
            I) arterial blood flow from neural excitation:   fa = N @ s
            II) balloon model:                               Y = A @ fa
            III) BOLD signal:                                b = B @ Y
                 total:                                      b = B @ A @ N @ s
            s = [D,T]
            fa= [D,T]
            Y = [3,K,D,T]
                Y[0,:,:,:] = flow
                Y[1,:,:,:] = volume
                Y[2,:,:,:] = q
            b = [D,T] '''
    def __defineOperators(self):
        p = getattr(self,'params')
        K, D, T = p.numCompartments, p.numDepths, p.nT
        if self.__opsFlag == [K,D,T]: return
        self.ops = {}
        # neural operator N
        self.ops['N'] = LinearOperator(
            shape   = (D*T, D*T),
            matvec  = self.__fun_N,
            rmatvec = self.__fun_N,
            matmat  = self.__fun_N,
            rmatmat = self.__fun_N,
            dtype=np.float   
        )
        #dottest(self.ops['N'], verb=True)  # check, that forward and adjoint operators are correct
        # balloon operator A 
        '''
        self.ops['A'] = LinearOperator(
            shape   = (3*D*T, 3*K*D*T),
            matvec  = self.__fun_A,
            rmatvec = self.__fun_A,
            matmat  = self.__fun_A,
            rmatmat = self.__fun_A,
            dtype=np.float   
        )
        # BOLD operator B
        self.ops['B'] = LinearOperator(
            shape   = (3*K*D*T, D*T),
            matvec  = self.__fun_B,
            rmatvec = self.__fun_B,
            matmat  = self.__fun_B,
            rmatmat = self.__fun_B,
            dtype=np.float   
        ) '''
        # set flag
        self.__opsFlag = [K,D,T]

    # =================================== OPERATOR FUNCTIONS ================================
    def __fun_N(self, s):
        p, n = getattr(self,'params'), getattr(self,'nparams')
        D, T, dt = p.numDepths, p.nT, p.dt
        ne, ni, vas = np.zeros([D]), np.zeros([D]), np.zeros([D])
        fa = np.ones([D,T])
        # sp.signal.convolve2d(A,B, 'valid')
        s = s.reshape([D,T])
        for t in range(1,T):
            ne_ = ne + dt * (n.sigma * ne - n.mu * ni + n.C * s[:,t])
            ni_ = ni + dt * n.lambd * (ne - ni)
            vas = vas + dt * (ne - n.c1 * vas)
            fa[:,t]  = fa[:,t-1] * np.exp(dt * (n.c2 * vas - n.c3 * (fa[:,t-1] - 1)) / fa[:,t-1])
            ne, ni = ne_, ni_
        return fa

    ''' Y[0,:,:,:] = flow
        Y[1,:,:,:] = volume
        Y[2,:,:,:] = q '''
    def __fun_A(self, fa):
        p, n, c = getattr(self,'params'), getattr(self,'nparams'), getattr(self,'consts')
        K, D, T, dt = p.numCompartments,  p.numDepths, p.nT, p.dt
        f, f_, v, v_, q = np.ones([K,D+1]), np.ones([K,D+1]), np.ones([K,D+1]), np.ones([K,D+1]), np.ones([K,D+1])
        Y = np.ones([3, K, D+1, T])  # need D+1 to have a deeper layer for deepest layer (factor is 0, though -> value is irrelevant)
        dv, m = np.zeros([K-1,D]), np.zeros([K-1,D])
        # initial conditions
        f[1,0:D] = fa[:,0] * p.F0[0,:]/p.F0[1,:]
        f[2,D-1]     = f[1,D-1    ] * p.F0[1,D-1]    /p.F0[2,D-1]
        f[2,D-2::-1] = f[1,D-2::-1] * p.F0[1,D-2::-1]/p.F0[2,D-2::-1] + f[2,D-1:0:-1,0] * p.F0[2,D-1:0:-1]/p.F0[2,D-2::-1]
        v[1:3,0:D] = np.power(f[1:3,0:D], p.alpha[1:3,0:D])
        # IVP
        for t in range(1,T):
            flowdir = (dv>=0).astype('int')
            # flow
            f_[0,0:D] = fa[:,t]
            f_[1:3,0:D] = c['D'][1:3,:,flowdir] * np.power(v[1:3,0:D], 1/p.alpha[1:3,:]) \
                        + c['E'][1:3,:,flowdir] * f[0:2,0:D] \
                        + c['G'][1:3,:,flowdir] * f[1:3,1:D+1] 
            # volume
            dv[1:3,:] = c['K'][1:3,:] * Y[0,1:3,0:D,  t-1] \
                      + c['L'][1:3,:] * Y[0,0:2,0:D,  t-1] \
                      + c['M'][1:3,:] * Y[0,1:3,1:D+1,t-1]
            Y[1,1:3,0:D,t] = Y[1,1:3,0:D,t-1] * np.exp(dt *  dv[1:3,:] / Y[1,1:3,0:D,t-1])
            # q
            m[1,:] = (Y[0,0,0:D,t-1] - 1) / (p.n[1,:] - 1)
            m[2,:] = Y[2,2,0:D,t] / Y[1,2,0:D,t]
            dq = c['K'][1:3,:] * Y[0,1:3,0:D,  t-1] * Y[2,1:3,0:D,t] / Y[1,1:3,0:D,t] \
               + c['L'][1:3,:] * Y[0,0:2,0:D,  t-1] * m[1:3,:] \
               + c['M'][1:3,:] * Y[0,1:3,1:D+1,t-1] * Y[2,1:3,1:D+1,t] / Y[1,1:3,1:D+1,t]
            Y[2,1:3,0:D,t] = Y[2,1:3,0:D,t-1] * np.exp(dt *  dq / Y[2,1:3,0:D,t-1])        
        return Y
    
    ''' Y[1,:,:,:] = volume
        Y[2,:,:,:] = q  '''
    def __fun_B(self, Y):
        p, c = getattr(self,'params'), getattr(self,'consts')
        K, D, T, = p.numCompartments,  p.numDepths, p.nT
        return np.sum( \
                c['A'].reshape(K,D,1).repeat(T,2) * (1-Y[2,:,0:D,:]) + \
                c['B'].reshape(K,D,1).repeat(T,2) * (1-Y[2,:,0:D,:]/Y[1,:,0:D,:]) + \
                c['C'].reshape(K,D,1).repeat(T,2) * (1-Y[1,:,0:D,:]) \
            , axis=0)  # sum over K

    # =================================== CALCULATE MODEL ================================
    def forwardModel(self, inputTL: Input_Timeline):
        K,D,T = inputTL.params.numCompartments, inputTL.params.numDepths, inputTL.params.nT
        self.__setKDT(K=K, D=D, T=T)
        self.__defineOperators()
        n = self.ops['N'] @ inputTL.stimulus.reshape([D*T])
        #a = self.ops['A'] @ n
        #b = self.ops['B'] @ a
        return n.reshape([D,T])

    def inverseModel(self, inputData, plots=None):
        D,T = self.params.numDepths, self.params.nT
        if inputData.shape != [D,T]: print('Error')
        self.__defineOperators()
        s = self.ops['N'].H @ inputData.reshape([D*T])
        # it appears N.H = N
        return s.reshape([D,T])
    







''' example: 
m,n = 10,10
x = np.random.rand(m)

funname = lambda X: np.sum(X, axis=0).reshape(1,-1).repeat(X.shape[0], axis=0)

def funname(X): return np.sum(X, axis=0).reshape(1,-1).repeat(X.shape[0], axis=0)
A = LinearOperator(
    shape   = (m,n),
    matvec  = funname,
    rmatvec = funname,
    matmat  = funname,
    rmatmat = funname,
    dtype=np.float   
)
print(A @ x)

    '''


    
