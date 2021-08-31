"""
@name:      Balloon
@author:    Sula Spiegel
@change:    06/08/2021

@summary:   Class to calculate Balloon Model 
            (flow, volume, oxygenation fraction as function of time and model parameters)
            Compartments: [arteriole, venule, vein]
            Matrix dimensions: [K, D, T]
"""

''' TODO: q, BOLD, log-normal
'''

import numpy as np
import math
from warnings import warn
from class_ModelParameters import Model_Parameters
from class_inputTimeline import Input_Timeline
from class_BOLD import BOLD
from class_BalloonPlots import Balloon_Plots

class Balloon:
# ---------------------------------  INIT  ------------------------------------------------
    def __init__(self, params: Model_Parameters, input_TimeLine: Input_Timeline):
        self.params = params
        self.inputTL = input_TimeLine

        self.__check_input()
        self.__get_priors()
        self.__get_balloon()
        self.bold = BOLD(self)
        self.plots = Balloon_Plots(self)

# ---------------------------------  PREPARATION  -----------------------------------------
    ''' __check_input: make sure, f_arteriole is given '''
    def __check_input(self):
        if not self.inputTL.available_input[self.inputTL.INDEX_FLOW]:
            raise Exception('Balloon model needs f_arteriole as input. Calculate neural model first or give arterial flow directly.')

    ''' __get_priors: get time invariant constants (denominator of flow ("flowscaling")) '''
    def __get_priors(self):
        self.flowscaling = np.empty([self.params.numCompartments, self.params.numDepths, 2])
        for k in range(self.params.VENULE, self.params.numCompartments):
            for d in range(0, self.params.numDepths):
                for flowdir in range(0,2):
                    self.flowscaling[k,d,flowdir] = \
                        self.params.F0[k,d] * self.params.vet[k,d,flowdir] \
                      + self.params.V0[k,d]
    
    ''' __init_matrices: initialize all matrices needed for balloon model calculation '''
    def __init_matrices(self):
        # use ones, since init values should be 1
        K = self.params.numCompartments
        D = self.params.numDepths
        T = self.params.N
        self.volume = np.ones([K, D, T])
        self.flow = np.ones([K, D, T])
        self.q = np.ones([K, D, T])

        # save change values for one iteration
        self.dv = np.zeros([K, D])  # need zero for visco-elastic time constant
        self.dq = np.empty([K, D])
    
    ''' __init_values: get initial values (t=0) '''
    def __init_values(self):
        # arterial inflow
        self.flow[self.params.ARTERIOLE,:,:] = self.inputTL.f_arteriole
        # steady-state conditions
        t = 0
        for k in range(self.params.VENULE, self.params.numCompartments):
            for d in range(self.params.numDepths-1, -1, -1):
                self.flow[k,d,t] = \
                        self.__getPreviousCompartmentFlow(k,d,t) \
                    +   self.__getDeeperLayerFlow(k,d,t)
                self.volume[k,d,t] = math.pow(self.flow[k,d,t], self.params.alpha[k,d])

    ''' __get_flowDir: find out, if inflation or deflation (for visco-elastic time constant) '''
    def __get_flowDir(self, k,d):
        if self.dv[k,d] >= 0: self.flowdir = self.params.INFLATION
        else: self.flowdir = self.params.DEFLATION
        return self.flowdir  # give possibility to remember it

 # ---------------------  CHECKS FOR TIME LINE CALCULATION --------------------------------
    ''' __isDeepestLayer: return true, if d is index of lowest layer '''
    def __isDeepestLayer(self, d):
        return d+1 == self.params.numDepths
    
    ''' __needDeeperLayers: only need to consider deeper layers for higher layers of veins '''
    def __needDeeperLayers(self, k,d):
        return k == self.params.VEIN and not self.__isDeepestLayer(d)

# ---------------------------------  CALCULATE FLOW  --------------------------------------
    ''' __getCurrentFlowVolume: get part of flow resulting from current, depth/compartment-specific volume '''
    def __getCurrentFlowVolume(self, k,d,t):
        return self.params.V0[k,d] * pow(self.volume[k,d,t], 1/self.params.alpha[k,d])
    
    ''' __getPreviousCompartmentFlowVolume: get part of flow volume coming from previous compartment '''
    def __getPreviousCompartmentFlowVolume(self, k,d,t):
        return self.params.vet[k,d,self.flowdir] * self.params.F0[self.params.VENULE,d] * self.flow[k-1,d,t]
    
    ''' __getDeeperLayerFlowVolume: get part of flow volume coming from deeper layers '''
    def __getDeeperLayerFlowVolume(self, k,d,t):
        if not self.__needDeeperLayers(k,d): return 0
        return self.params.vet[k,d,self.flowdir] * self.params.F0[k,d+1] * self.flow[k,d+1,t]
    
    ''' __get_flow: get flow for vein or venule from volume (and write into time line) '''
    def __get_flow(self, k,d,t):
        self.flow[k,d,t] = \
            (   self.__getCurrentFlowVolume(k,d,t) \
              + self.__getPreviousCompartmentFlowVolume(k,d,t) \
              + self.__getDeeperLayerFlowVolume(k,d,t) \
            ) / self.flowscaling[k,d,self.flowdir]
            # deeperLayer only for vein

# ---------------------------------  CALCULATE VOLUME CHANGE  -----------------------------
    ''' __getCurrentFlow: get flow resulting from current depth/compartment '''
    def __getCurrentFlow(self, k,d,t):
        return self.flow[k,d,t]
    
    ''' __getPreviousCompartmentFlow: get flow resulting from previous compartment '''
    def __getPreviousCompartmentFlow(self, k,d,t):
        return self.__getCurrentFlow(k-1,d,t) * self.params.F0[k-1,d] / self.params.F0[k,d]
    
    ''' __getDeeperLayerFlow: get part of flow coming from deeper layers '''
    def __getDeeperLayerFlow(self, k,d,t):
        if not self.__needDeeperLayers(k,d): return 0
        return self.__getCurrentFlow(k,d+1,t) * self.params.F0[k,d+1] / self.params.F0[k,d]

    ''' __get_dv: get volume change for vein or venule from flow (and write into time line) '''
    def __get_dv(self, k,d,t):
        self.dv[k,d] = \
            (     self.__getPreviousCompartmentFlow(k,d,t) \
                + self.__getDeeperLayerFlow(k,d,t) \
                - self.__getCurrentFlow(k,d,t) \
            ) / self.params.tau0[k,d]
    
# ---------------------------------  CALCULATE DEOXY CHANGE  ------------------------------
    ''' __getCurrentHbV: get Hb/v in current depth/compartment '''
    def __getCurrentHbV(self, k,d,t):
        return self.q[k,d,t] / self.volume[k,d,t]
    
    ''' __getPreviousCompartmentHbV: get Hb/v in previous compartment '''
    def __getPreviousCompartmentHbV(self, k,d,t):
        if k == self.params.VENULE:
            fa = self.flow[self.params.ARTERIOLE,d,t]
            return (fa + self.params.n - 1) / (self.params.n * fa)  # divide by fa because it'll be multiplied in dq 
        return self.__getCurrentHbV(k-1,d,t)
    
    ''' __getDeeperLayerHbV: get Hb/v from deeper layer '''
    def __getDeeperLayerHbV(self, k,d,t):
        if not self.__needDeeperLayers(k,d): return 0
        return self.__getCurrentHbV(k,d+1,t)

    ''' __get_dq: get change of Hb-val for venule or vein from flow (and write into time line) '''
    def __get_dq(self, k,d,t):
        self.dq[k,d] = \
            (     self.__getPreviousCompartmentFlow(k,d,t) * self.__getPreviousCompartmentHbV(k,d,t) \
                + self.__getDeeperLayerFlow(k,d,t) * self.__getDeeperLayerHbV(k,d,t) \
                - self.__getCurrentFlow(k,d,t) * self.__getCurrentHbV(k,d,t) \
            ) / self.params.tau0[k,d]
        # deeperLayer only for vein
    
# ----------------------------- HELPERS FOR MODEL CALCULATION  ----------------------------
    ''' __get_balloonVal: general call function to get a single value of the time line;
                          assigns specific function that is to be executed '''
    def __get_balloonVal(self, k,d,t, varname):
        if varname is 'flow': self.__get_flow(k,d,t)
        elif varname is 'dv': self.__get_dv(k,d,t)
        elif varname is 'dq': self.__get_dq(k,d,t)
        else: warn(f"__get_balloonVal({varname}) not implemented -> ignoring call")
    
    ''' __get_oneLayer: get all balloon values for one layer/compartment/time point 
                        * assign flowdir (in-/deflation) in dependence of last dv
                        * at the end, check that correct flowdir was used and repeat if not
                        * use <firstcall> to repeat only once 
                            (make sure flowdir doesn't swap forth and back infinitely) '''
    def __get_oneLayer(self, k,d,t, firstcall=True):
        # define variables that are needed and order to calculate them
        required_vals = ["flow", "dv", "dq"]
        # get (and save) flow dir from last volume change
        flowdir_tmp = self.__get_flowDir(k,d)
        # get required variables
        for v in range(len(required_vals)):
            self.__get_balloonVal(k,d,t, required_vals[v])
        # check, that for this time point flow dir didn't change
        if self.__get_flowDir(k,d) != flowdir_tmp and firstcall:
            # if it did change, repeat with correct visco-elastic time constant
            self.__get_oneLayer(k,d,t, False)
    
    ''' __newTimePoint: get next point on time line (for v,q) by log-normal-transformation '''
    def __newTimePoint(self, oldval, d_val):
        return oldval * math.exp(self.params.dt * d_val / oldval)
    
    ''' __update_timePoints: write values for next point on time line of v,q into time lines '''
    def __update_timePoints(self, t):
        for k in range(self.params.VENULE, self.params.numCompartments):
            for d in range(0, self.params.numDepths):
                self.volume[k,d,t+1] = self.__newTimePoint(self.volume[k,d,t], self.dv[k,d])
                self.q[k,d,t+1] = self.__newTimePoint(self.q[k,d,t], self.dq[k,d])
    
# ------------------------------------- MAIN MODEL CALCULATION  ---------------------------
    ''' __get_balloon: use all balloon equations (for all time points) '''
    def __get_balloon(self):
        # init values at start
        self.__init_matrices()  # flow, v, q are set to 1, dv is set to 0
        self.__init_values()  # make sure steady-state conditions are met
        
        # go through time course
        for t in range(0, self.params.N - 1):
            for k in range(self.params.VENULE, self.params.numCompartments):
                for d in range(self.params.numDepths-1, -1, -1):
                    self.__get_oneLayer(k,d,t)
            self.__update_timePoints(t)
    


# ------------------------------------- DEPRECEATED  -------------------------------------
    '''__get_extractionFraction: get the extraction fraction for entire time flow 
    # todo: is this relation valid in the new model?
    def __get_extractionFraction(self, params: Model_Parameters):
        self.E = np.empty([params.numDepths, params.N])
        for d in range(0, params.numDepths):
            for t in range(0, params.N):
                self.E[d,t] = 1 - pow((1 - params.E0[0]), (1/self.flow[0,d,t]))

    __get_flow_from_volume: get venous outflow from venous volume
    def __get_flow_from_volume(self, params: Model_Parameters, k, d, t, dv):
        if dv > 0: tau_v = params.vet[k,d,0]  # inflation
        else: tau_v = params.vet[k,d,1]  # deflation
        self.flow[k,d,t] = self.volume[k,d,t]^(1/params.alpha[k,d]) + tau_v * dv

    __get_balloon: use all balloon equations (for all time points)
    def __get_balloon(self, params: Model_Parameters):
        # init values at start
        for d in range(params.numDepths-1, -1):  # start at lowest level (highest d)
            # init flow
            self.flow[1,d,0] = self.flow[0,d,0]  # assume steady-state at init -> venule = arteriole
            self.flow[2,d,0] = self.flow[1,d,0] * params.F0[1,d] / params.F0[2,d]  # veins collect venules
            if d < params.numDepths-1:
                self.flow[2,d,0] += self.flow[2,d+1,0] * params.F0[2,d+1] / params.F0[2,d]  # add lower vein if not lowest depth
            # init volume
            for k in range(0, params.numCompartments):
                self.volume[k,d,0] = self.flow[k,d,0]^params.alpha[k,d]
            # init q
            self.q[1,d,0] = self.volume[1,d,0] * self.E[d,0] / params.E0[1]
            self.q[2,d,0] = self.flow[1,d,0] * params.F0[1,d] / params.F0[2,d] * self.E[d,0] / params.E0[2]
            if d < params.numDepths-1:
                self.q[2,d,0] += self.flow[2,d+1,0] * params.F0[1,d] / params.F0[2,d] * self.q[2,d+1,0]/self.volume[2,d+1,0]
            self.q[2,d,0] *= self.volume[2,d,0] / self.flow[2,d,0]
        
        # go through time course
        dv = np.empty([params.numCompartments])
        dq = np.empty([params.numCompartments])
        for t in range(0, params.N):
            # change parameters
            for d in range(params.numDepths-1, -1, -1):
                dv[1] = self.flow[0,d,t] - self.flow[1,d,t]   # venule
                dq[1] = self.flow[0,d,t] * self.E[d,t] / params.E0[1] - self.flow[1,d,t] * self.q[1,d,t] / self.volume[1,d,t]

                dv[2] = self.flow[1,d,t] * params.F0[1,d] / params.F0[2,d] - self.flow[2,d,t]  # vein
                dq[2] = self.flow[1,d,t] * params.F0[1,d] / params.F0[2,d] * self.q[1,d,t]/self.volume[1,d,t] \
                        - self.flow[2,d,t] * self.q[2,d,t]/self.volume[2,d,t]
                
                if d < params.numDepths-1:
                    dv[2] += self.flow[2,d+1,t] * params.F0[2,d+1] / params.F0[2,d] 
                    dq[2] += self.flow[2,d+1,t] * params.F0[2,d+1] / params.F0[2,d] * self.q[2,d+1,t]/self.volume[2,d+1,t]

                for k in range(1, params.numCompartments):
                    dv[k] /= self.tau0[k,d]
                    dq[k] /= self.tau0[k,d]
                
            # new parameters
            for k in range(1, params.numCompartments):
                for d in range(params.numDepths-1, -1):
                    self.volume[k,d,t] += dv[k]
                    self.q[k,d,t] += dq[k]
                    self.__get_flow_from_volume(params, k, d, t, dv[k])
                    '''