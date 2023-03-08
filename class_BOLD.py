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
        E0_is_boldparam = 'E0' in self.params.boldparams
        # ci
        self.consts = {'c': np.empty([3, self.params.numCompartments]),
                       'H0': np.empty([self.params.numDepths]),
                       'sV0': np.zeros([self.params.numDepths])}
        for k in range(0, self.params.numCompartments):
            if E0_is_boldparam: e0 = self.params.boldparams['E0'][k]
            else: e0 = self.params.E0[k,:]
            self.consts['c'][0,k] = 4.3 * \
                self.params.boldparams['dXi'] * \
                self.params.boldparams['Hct'][k] * \
                self.params.boldparams['gamma0'] * \
                self.params.boldparams['B0'] * \
                e0 * \
                self.params.boldparams['TE'] / \
                (2*np.pi)
            self.consts['c'][1,k] = \
                self.params.boldparams['epsilon'][k] * \
                self.params.boldparams['r0'][k] * \
                e0 * \
                self.params.boldparams['TE']
            self.consts['c'][2,k] = 1 - self.params.boldparams['epsilon'][k]
        # H0, sum of V0
        for d in range(0, self.params.numDepths):
            sev = 0  # sum of epsilon * V0
            for k in range(0, self.params.numCompartments):
                v = self.params.V0[k,d] * self.params.numDepths / 100  # transformation from total volume to blood volume fraction
                self.consts['sV0'][d] += v  # has been initialized as 0
                sev += v * self.params.boldparams['epsilon'][k]
            self.consts['H0'][d] = 1 / (1 - self.consts['sV0'][d] + sev)
    
    ''' __get_BOLD: '''
    def __get_BOLD(self):
        self.BOLDsignal = np.empty([self.params.numDepths, self.params.N])
        self.VASOsignal = np.empty([self.params.numDepths, self.params.N])
        for t in range(0, self.params.N):
            for d in range(0, self.params.numDepths):
                B1 = 0  # extra-vascular
                B2 = 0  # intra-vascular
                B3 = 0  # volume change
                VA = 0  # VASO-signal 
                # todo: weightings of different compartments (None, according to Havlicek)
                for k in range(0, self.params.numCompartments):
                    v = self.params.V0[k,d] * self.params.numDepths / 100  # transformation from total volume to blood volume fraction
                    B1 += self.consts['c'][0,k] * v * (1-self.parent.q[k,d,t])
                    B2 += self.consts['c'][1,k] * v * (1-self.parent.q[k,d,t]/self.parent.volume[k,d,t])
                    B3 += self.consts['c'][2,k] * v * (1-self.parent.volume[k,d,t])
                    VA += (self.parent.volume[k,d,t] - self.params.V0[k,d]) / (self.params.V0[k,d] - 1)  # [ml blood/100ml tissue]
                B1 *= (1 - self.consts['sV0'][d])
                self.BOLDsignal[d,t] = self.consts['H0'][d] * (B1 + B2 + B3)
                self.VASOsignal[d,t] = VA
  