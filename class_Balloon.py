import numpy as np
from class_Input import Input
from class_BOLD import BOLD
from class_BalloonPlots import BalloonPlots

DEBUG = 1
def msg(txt):
    if DEBUG: print(txt)

class Balloon:
    def __init__(self, input: Input):
        self.__init2(input)
    
    def __init2(self, input: Input):
        self.input = input
        self.__init_matrices(input)
        self.__get_tau0(input)
        self.__get_extractionFraction(input)
        self.__get_balloon(input)
        self.bold = BOLD(self)
        self.plots = BalloonPlots(self)

    def __init_matrices(self, input: Input):
        self.volume = np.empty([input.numCompartments, input.numDepths, input.N])
        self.flow = np.empty([input.numCompartments, input.numDepths, input.N])  # arteriole, venule, vein
        self.q = np.zeros([input.numCompartments, input.numDepths, input.N])  # no extraction in arterioles

        self.flow[0, :, :] = input.f_arteriole

    # get time constant for flow (tau0 = V0/F0)
    def __get_tau0(self, input: Input):
        self.tau0 = np.empty([input.numCompartments, input.numDepths])
        for i in range(1, input.numCompartments):
            for k in range(0, input.numDepths):
                self.tau0[i,k] = input.V0[i,k]/input.F0[i,k]
    
    # get extraction fraction
    # todo: is this relation valid in the new model?
    def __get_extractionFraction(self, input: Input):
        self.E = np.empty([input.numDepths, input.N])
        for k in range(0, input.numDepths):
            for t in range(0, input.N):
                self.E[k,t] = 1 - pow((1 - input.E0[0]), (1/input.f_arteriole[0,k,t]))

    def __get_flow_from_volume(self, input: Input, i: int, k: int, t:int, dv):
        if dv > 0:
            tau_v = input.vet[i,k,0]  # inflation
        else:
            tau_v = input.vet[i,k,1]  # deflation
        self.flow[i,k,t] = self.volume[i,k,t]^(1/input.alpha[i,k]) + tau_v * dv
        return self.flow[i,k,t]

    # get all balloon equations
    def __get_balloon(self, input: Input):
        # init values at start
        for k in range(input.numDepths-1, -1):  # start at lowest level (highest k)
            # init flow
            self.flow[1,k,0] = self.flow[0,k,0]  # assume steady-state at init -> venule = arteriole
            self.flow[2,k,0] = self.flow[1,k,0] * input.F0[1,k] / input.F0[2,k]  # veins collect venules
            if k < input.numDepths-1:
                self.flow[2,k,0] += self.flow[2,k+1,0] * input.F0[2,k+1] / input.F0[2,k]  # add lower vein if not lowest depth
            # init volume
            for i in range(0, input.numCompartments):
                self.volume[i,k,0] = self.flow[i,k,0]^input.alpha[i,k]
            # init q
            self.q[1,k,0] = self.volume[1,k,0] * self.E[k,0] / input.E0[1]
            self.q[2,k,0] = self.flow[1,k,0] * input.F0[1,k] / input.F0[2,k] * self.E[k,0] / input.E0[2]
            if k < input.numDepths-1:
                self.q[2,k,0] += self.flow[2,k+1,0] * input.F0[1,k] / input.F0[2,k] * self.q[2,k+1,0]/self.volume[2,k+1,0]
            self.q[2,k,0] *= self.volume[2,k,0] / self.flow[2,k,0]
        
        # go through time course
        dv = np.empty([input.numCompartments])
        dq = np.empty([input.numCompartments])
        for t in range(0, input.N):
            # change parameters
            for k in range(input.numDepths-1, -1):
                dv[1] = self.flow[0,k,t] - self.flow[1,k,t]   # venule
                dq[1] = self.flow[0,k,t] * self.E[k,t] / input.E0[1] - self.flow[1,k,t] * self.q[1,k,t] / self.volume[1,k,t]

                dv[2] = self.flow[1,k,t] * input.F0[1,k] / input.F0[2,k] - self.flow[2,k,t]  # vein
                dq[2] = self.flow[1,k,t] * input.F0[1,k] / input.F0[2,k] * self.q[1,k,t]/self.volume[1,k,t] \
                        - self.flow[2,k,t] * self.q[2,k,t]/self.volume[2,k,t]
                
                if k < input.numDepths-1:
                    dv[2] += self.flow[2,k+1,t] * input.F0[2,k+1] / input.F0[2,k] 
                    dq[2] += self.flow[2,k+1,t] * input.F0[2,k+1] / input.F0[2,k] * self.q[2,k+1,t]/self.volume[2,k+1,t]

                for i in range(1, input.numCompartments):
                    dv[i] /= self.tau0[i,k]
                    dq[i] /= self.tau0[i,k]
                
            # new parameters
            for i in range(1, input.numCompartments):
                for k in range(input.numDepths-1, -1):
                    self.volume[i,k,t] += dv[i]
                    self.q[i,k,t] += dq[i]
                    self.__get_flow_from_volume(input, i, k, t, dv[i])
    
    def reset_input(self, new_input: Input):
        self.__init2(new_input)
    
    def reset_fArteriole(self, new_f):
        self.input.set_fArteriole(new_f)
        self.__init2(self.input)