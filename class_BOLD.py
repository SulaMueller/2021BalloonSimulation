"""
@name:      BOLD
@author:    Sula Spiegel
@change:    10/08/2021

@summary:   get BOLD-signal from BALLOON
"""

import numpy as np

class BOLD:
    def __init__(self, parent):
        self.parent = parent
        self.params = parent.params
        self.__get_scalingConstants()
        self.__get_BOLD()
    
    ''' __get_scalingConstants: '''
    def __get_scalingConstants(self):
        # ci
        self.consts = {'cs': np.empty([3, self.params.numCompartments]),
                       'H0': np.empty([self.params.numDepths]),
                       'sV0': np.zeros([self.params.numDepths])}
        for k in range(0, self.params.numCompartments):
            self.consts['cs'][0,k] = 4.3 * self.params.boldparams['dXi'] * self.params.boldparams['Hct'][k] * self.params.boldparams['gamma0'] * self.params.B0 \
                                 * self.params.E0[k] * self.params.boldparams['TE']
            self.consts['cs'][1,k] = self.params.boldparams['epsilon'][k] * self.params.boldparams['r0'][k] * self.params.E0[k] * self.params.boldparams['TE']
            self.consts['cs'][2,k] = 1-self.params.boldparams['epsilon'][k]
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
                    B1 += self.consts['cs'][0,k] * self.params.V0[k,d] * (1-self.parent.q[k,d,t])
                    B2 += self.consts['cs'][1,k] * self.params.V0[k,d] * (1-self.parent.q[k,d,t]/self.parent.volume[k,d,t])
                    B3 += self.consts['cs'][2,k] * self.params.V0[k,d] * (1-self.parent.volume[k,d,t])
                B1 *= (1 - self.consts['sV0'][d])
                self.BOLDsignal[d,t] = self.consts['H0'][d] * (B1 + B2 + B3)
                self.VASOsignal[d,t] = self.consts['H0'][d] * B3
  