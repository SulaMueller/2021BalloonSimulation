import numpy as np

gamma0 = 2*np.pi*42.58*pow(10,6)

class BOLD:
    def __init__(self, parent):
        self.parent = parent
        self.input = parent.input
        self.__get_scalingConstants()
        self.__get_BOLD()
    
    def __get_scalingConstants(self):
        # ci
        self.consts = {'cs': np.empty([3, self.input.numCompartments]),
                       'H0': np.empty([self.input.numDepths]),
                       'sV0': np.zeros([self.input.numDepths])}
        for i in range(0, self.input.numCompartments):
            self.consts['cs'][0,i] = 4.3 * self.input.boldparams['dXi'] * self.input.boldparams['Hct'][i] * gamma0 * self.input.B0 \
                                 * self.input.E0[i] * self.input.boldparams['TE']
            self.consts['cs'][1,i] = self.input.boldparams['epsilon'][i] * self.input.boldparams['r0'][i] * self.input.E0[i] * self.input.boldparams['TE']
            self.consts['cs'][2,i] = 1-self.input.boldparams['epsilon'][i]
        # H0, sum of V0
        for k in range(0, self.input.numDepths):
            sev = 0  # sum of epsilon * V0
            for i in range(0, self.input.numCompartments):
                self.consts['sV0'][k] += self.input.V0[i,k]
                sev += self.input.V0[i,k] * self.input.boldparams['epsilon'][i]
            self.consts['H0'][k] = 1 / (1 - self.consts['sV0'][i] + sev)
    
    def __get_BOLD(self):
        self.BOLDsignal = np.empty([self.input.numDepths, self.input.N])
        self.VASOsignal = np.empty([self.input.numDepths, self.input.N])
        for t in range(0, self.input.N):
            for k in range(0, self.input.numDepths):
                B1 = 0
                B2 = 0
                B3 = 0
                for i in range(0, self.input.numCompartments):
                    B1 += self.consts['cs'][0,i] * self.input.V0[i,k] * (1-self.parent.q[i,k,t])
                    B2 += self.consts['cs'][1,i] * self.input.V0[i,k] * (1-self.parent.q[i,k,t]/self.parent.volume[i,k,t])
                    B3 += self.consts['cs'][2,i] * self.input.V0[i,k] * (1-self.parent.volume[i,k,t])
                B1 *= (1 - self.consts['sV0'][k])
                self.BOLDsignal[k,t] = self.consts['H0'][k] * (B1 + B2 + B3)
                self.VASOsignal[k,t] = self.consts['H0'][k] * B3
  