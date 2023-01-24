"""
@name:      BOLD
@author:    Sula Spiegel
@change:    10/08/2021

@summary:   get BOLD-signal from BALLOON (also gives VASO-signal)
"""

import numpy as np
from class_ModelParameters import clearAttrs

class BOLD:
    def __init__(self, parent):
        self.parent = parent  # parent: Balloon
        self.params = parent.params
        self.__get_scalingConstants()
        self.__get_BOLD()
        clearAttrs(self, ['consts'])
    
    ''' __get_scalingConstants: '''
    def __get_scalingConstants(self):
        E0_not_depthDependent = hasattr(self.params.boldparams, 'E0')
        # ci
        self.consts = {'cs': np.empty([3, self.params.numCompartments, self.params.numDepths]),
                       'H0': np.empty([self.params.numDepths]),
                       'sV0': np.zeros([self.params.numDepths])}
        for k in range(0, self.params.numCompartments):
            if E0_not_depthDependent: e0 = self.params.boldparams['E0'][k]
            else: e0 = self.params.E0[k,:]
            self.consts['cs'][0,k,:] = 4.3 * \
                self.params.boldparams['dXi'] * \
                self.params.boldparams['Hct'][k] * \
                self.params.boldparams['gamma0'] * \
                self.params.boldparams['B0'] * \
                e0 * \
                self.params.boldparams['TE']
            self.consts['cs'][1,k,:] = \
                self.params.boldparams['epsilon'][k] * \
                self.params.boldparams['r0'][k] * \
                e0 * \
                self.params.boldparams['TE']
            self.consts['cs'][2,k,:] = 1 - self.params.boldparams['epsilon'][k]
        # H0, sum of V0
        for d in range(0, self.params.numDepths):
            sev = 0  # sum of epsilon * V0
            for k in range(0, self.params.numCompartments):
                self.consts['sV0'][d] += self.params.V0[k,d]
                sev += self.params.V0[k,d] * self.params.boldparams['epsilon'][k]
            self.consts['H0'][d] = 1 / (1 - self.consts['sV0'][k] + sev)
    
    ''' __get_BOLD: '''
    def __get_BOLD(self):
        self.BOLDsignal = np.empty([self.params.numDepths, self.params.N])
        self.VASOsignal = np.empty([self.params.numDepths, self.params.N])
        for t in range(0, self.params.N):
            for d in range(0, self.params.numDepths):
                B1 = 0
                B2 = 0
                B3 = 0
                for k in range(0, self.params.numCompartments):
                    B1 += self.consts['cs'][0,k,d] * self.params.V0[k,d] * (1-self.parent.q[k,d,t])
                    B2 += self.consts['cs'][1,k,d] * self.params.V0[k,d] * (1-self.parent.q[k,d,t]/self.parent.volume[k,d,t])
                    B3 += self.consts['cs'][2,k,d] * self.params.V0[k,d] * (1-self.parent.volume[k,d,t])
                B1 *= (1 - self.consts['sV0'][d])
                self.BOLDsignal[d,t] = self.consts['H0'][d] * (B1 + B2 + B3)
                self.VASOsignal[d,t] = self.consts['H0'][d] * B3
  