
import copy
import numpy as np
import math
import matplotlib.pyplot as plt

from warnUsr import warn
from readFile import readFiles
from plotHDM import newfig, getTime, plot1D, plotDepthwise, plotManyFunctions
from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters
from ivp import hemodynamic_model_forward

def rmse(y, data): return np.sqrt(((y - data) ** 2).mean())
def int2bin(pos_int, numDigits=None):  # transform an integer into an array of booleans (eg int2bin(11) = [1 0 1 1])
    if numDigits is None: 
        if pos_int==0: numDigits = 1
        else: numDigits = np.int32(np.floor(math.log2(pos_int))) + 1
    bin_array = np.zeros(numDigits, dtype=bool)
    for i in range(0, numDigits):
        bin_array[numDigits-i-1] = pos_int % 2 > 0
        pos_int = pos_int // 2
    return np.int32(bin_array)
def upsample(lo, numPoints_hi): 
    numPoints_hi = np.int32(numPoints_hi)
    repFactor = np.int32(np.round(numPoints_hi/len(lo)))
    while repFactor * len(lo) < numPoints_hi: repFactor = repFactor+1
    if lo.ndim==1: return np.repeat(lo, repFactor)[0:numPoints_hi]
    if lo.ndim==2: warn('ToDo: finish upsampling!')
def downsample(hi, numPoints_lo):  # take the point from the middle of each slice -> ok, since data is usually not downsampled (only model, stimuli)
    numPoints_lo = np.int32(numPoints_lo)
    sliceDist = np.int32(hi.shape[-1]/numPoints_lo)
    if hi.ndim==1: return hi[np.int32(sliceDist/2)::sliceDist][0:numPoints_lo]
    if hi.ndim==2: return hi[:,np.int32(sliceDist/2)::sliceDist][:,0:numPoints_lo]

''' MatrixParameters: define scope of models '''
class Matrix_Parameters:
    def __init__(self, dt, nT, D, require_nT=True):
        self.dt = dt
        self.nT = np.int32(nT)
        self.D = np.int32(D)
        self.require_nT = require_nT  # eg forward model doesn't need nT (gets it from stimulus length)
    
    def latchOnModelParams(self, modelparams:Model_Parameters):
        newparams = copy.deepcopy(modelparams)
        newparams.dt = self.dt
        newparams.nT = self.nT
        newparams.numDepths = self.D
        return newparams

''' aPriori: give some a priori info about the stimulus '''
class aPriori:
    def __init__(self, maxDelay, minDelay=0, depthDependency=True, minStimulusLength=1, maxStimulusLength=None, minDistBetweenStimuli=None):
        self.minDelay = minDelay  # [s]; time after which most of the effect starts
        self.maxDelay = maxDelay  # [s]; time after which most of the effect has happened (both define the time window where the effect is calculated)
        self.depthDependency = depthDependency  # bool
        self.minStimulusLength = minStimulusLength  # [s]
        self.maxStimulusLength = maxStimulusLength  # [s]
        if minDistBetweenStimuli is None: self.minDistBetweenStimuli = self.minStimulusLength  # [s]
        else: self.minDistBetweenStimuli = minDistBetweenStimuli  # [s]
        self.__defineProperties()
        
    def __defineProperties(self):
        self.unit = 's'
        self.type = float
        self.unitDependent = ['minDelay', 'maxDelay', 'minStimulusLength', 'maxStimulusLength', 'minDistBetweenStimuli']
        self.bools = ['depthDependency']
    
    def set(self, attrname, value):
        if attrname in self.unitDependent and type(value) == self.type:
            setattr(self, attrname, value)
        elif attrname in self.bools and value in [0,1] or type(value) == bool:
            setattr(self, attrname, bool(value))
        elif attrname in self.unitDependent and self.type == float and type(value) == int:
            setattr(self, attrname, value)
        else: warn(f'aPriori-class, set({attrname}) needs {self.properties["datatypes"][attrname]} as input.')
    
    def convert(self, conversionFactor, unitName):  # conversionFactor = s/newUnit
        converted = aPriori(self.maxDelay*conversionFactor, self.minDelay*conversionFactor, self.depthDependency, self.minStimulusLength*conversionFactor, self.maxStimulusLength, self.minDistBetweenStimuli*conversionFactor)
        if self.maxStimulusLength is not None: converted.maxStimulusLength = converted.maxStimulusLength * conversionFactor
        converted.unit = unitName
        return converted

''' hemodynamic_model_inverse: class to compute the inverse of the hemodynamic model -> best guess of stimulus that fits given (signal) data
        * data should have the shape [D,T]
        * stimulus should be [T]'''
class hemodynamic_model_inverse:
    def __init__(self, 
                 params: Model_Parameters, \
                 neural_params: Neural_Parameters):
        self.params, self.neural_params  = params, neural_params
        self.setSensitivity('round')
        self.verbose = True
    
    def setSensitivity(self, sensitivity='low'): self.sensitivity = sensitivity  # define, if estimate is floor'd ('low'), ceil'd ('high') or rounded
    def calculateForwardModel(self, stimulus1D_hi_res, y0=None, method='ivp'):
        if method=='ivp' and not hasattr(self, 'IVP'): self.__initIVP(self.params.dt, self.params.nT, self.params.numDepths)
        if method=='ivp': _, y, s = self.IVP.solveIVP(stimulus1D_hi_res, y0)
        if y0 is None: return s
        return s, y[:,-1]

    ## =================== HELPERS FOR ESTIMATORS ====================
    ## ---------------- INIT -----------------
    def __initIVP(self, dt, nT, D):
        self.p_model = Matrix_Parameters(dt, nT, D, require_nT=False)  # parameters for forward model
        self.IVP = hemodynamic_model_forward(self.p_model.latchOnModelParams(self.params), self.neural_params)
    def __setModelParams(self, dt, nT, D):
        if hasattr(self, 'p_model') and hasattr(self, 'IVP'):
            if self.p_model.D != D: return self.__initIVP(dt, nT, D)
            self.p_model.dt, self.p_model.nT, self.IVP.params.dt = dt, np.int32(nT), dt
        else: self.__initIVP(dt, nT, D)
    def __getMatrixParams(self, data, aPriori:aPriori, numTestPoints, dt_data):
        # get matrix parameters for data
        if dt_data is None: dt_data = self.params.boldparams['TR']
        self.p_data = Matrix_Parameters(dt_data, nT=data.shape[1], D=data.shape[0], require_nT=True)  # parameters for data
        T_total = self.p_data.dt * self.p_data.nT + self.p_data.dt  # data points are defined as in middle of data slice
        # get matrix parameters for partial and total stimuli (in coarse estimation)
        dt_S = aPriori.minStimulusLength
        T_vary = min(dt_S*numTestPoints, T_total)
        T_test = min(T_vary + aPriori.maxDelay, T_total)  # test stimulus = test points + zeros during delay
        if T_test < 2 * self.p_data.dt: 
            return warn('T_test too small. Should cover at least two data points. Increase numTestPoints. ')
        nT_S_test = np.int32(np.ceil(T_test/dt_S))
        T_test = nT_S_test * dt_S  # correct for overtime
        self.p_S_test = Matrix_Parameters(dt=dt_S, nT=nT_S_test, D=self.p_data.D, require_nT=True)  # parameters for the test stimulus
        nT_S_total = np.int32(np.ceil(T_total/dt_S))
        self.p_S_total = Matrix_Parameters(dt=dt_S, nT=nT_S_total, D=self.p_data.D, require_nT=True)  # parameters for the entire stimulus
        # get model parameters for test window (partial model)
        self.__setModelParams(dt=self.params.dt, nT=T_test/self.params.dt, D=self.p_data.D)  # create self.p_model and self.IVP with fitting D, dt
        return T_total, T_test, T_vary
    
    ## ---------------- STIMULUS COMBINATIONS -----------------
    def __getCombinations(self, numWindowPoints):  # get all possible combinations of stimulus values in a given window size (stimulus can be 0 or 1)
        numCombinations = np.int32(2**numWindowPoints)
        combinations = np.zeros([numCombinations, numWindowPoints])
        for i in range(0, numCombinations): combinations[i, :] = int2bin(i,numWindowPoints)
        return combinations
    def __comboMeetsApriori_inside(self, combo, aPriori:aPriori):
        if len(combo) > 2: 
            warn('implement __comboMeetsApriori_inside. Not implemented at the moment.')
            return False
        return True
    def __comboMeetsApriori(self, combo, aPriori__S:aPriori, S_estimate, firstTestPoint):
        # define required functions
        def __ind4getPart(len_vec, len_part, wanted='first'):
            if wanted=='first': return len_part
            else: return len_vec-1-len_part
        def __getPart(vec, wanted='first'):
            len_part = 0
            if wanted=='first': val = vec[0]
            else: val = vec[-1]
            while vec[__ind4getPart(len(vec), len_part, wanted)] == val:
                len_part = len_part+1
                if len_part == len(vec): break
            return len_part, val
        # do the magic
        if not np.any(S_estimate>0): return True
        len_S, sVal = __getPart(S_estimate[0:firstTestPoint], 'last')
        len_Combo, comboVal = __getPart(combo, 'first')
        if sVal == 1 and comboVal == 0 and len_S < aPriori__S.minStimulusLength: return False  # check for minStimulusLength
        elif sVal == 1 and comboVal == 1:
            if len_S + len_Combo < aPriori__S.minStimulusLength and len_Combo < len(combo): return False  # check for minStimulusLength
            if aPriori__S.maxStimulusLength is not None:
                if len_S + len_Combo > aPriori__S.maxStimulusLength: return False  # check for maxStimulusLength
        elif sVal == 0 and comboVal == 1:  # check for minDistBetweenStimuli
            if len_S < aPriori__S.minDistBetweenStimuli and len_S > len(S_estimate): return False
        return True
    
    ## ---------------- FIT ERROR -----------------
    def __getExactTimePoint_inData(self, i_data): return i_data*self.p_data.dt + self.p_data.dt/2  # define them at the middle of the slice
    def __getDataIndex(self, t): 
        i = min(self.p_data.nT-1, max(0, np.int32(np.round(t/self.p_data.dt)) )) # round will switch in the middle of the slice
        return i, self.__getExactTimePoint_inData(i)  # return i, t : index and corresponding exact time point
    def __getTimeAxisOfData(self, firstTestPoint, T_test, T_total):
        t_test_start = firstTestPoint * self.p_S_test.dt
        t_test_end = min(t_test_start + T_test, T_total)
        i_data_start, t_test_start = self.__getDataIndex(t_test_start)
        i_data_end, t_test_end = self.__getDataIndex(t_test_end)
        if t_test_end - t_test_start + self.p_data.dt/2 > T_test:
            i_data_end = i_data_end-1
            t_test_end = self.__getExactTimePoint_inData(i_data_end)
        return np.linspace(t_test_start, t_test_end, i_data_end - i_data_start + 1), i_data_start, i_data_end
    def __convertDataAxisToModelIndice(self, timeaxis_data):  # get time points in model that are closest to corresponding data points on timeaxis_data
        t0 = timeaxis_data[0] - self.p_data.dt/2
        indice_model = np.zeros(len(timeaxis_data), int)  # can't just slice, since step could be irregular (dt_data is not an exact multiple of dt_model)
        for i, t in enumerate(timeaxis_data): indice_model[i] = np.int32(np.round((t-t0)/self.p_model.dt))
        return indice_model
    def __getFitError(self, signal_estimate_partial, data_partial, errorType='rmse'): 
        if errorType=='rmse': return rmse(signal_estimate_partial, data_partial)

    ## ---------------- MISC -----------------
    def __averagePoint(self, estimatedStimuli_tmp, firstRow, lastRow, isEnd=False):
        # estimatedStimuli_tmp = identical points are diagonal ("firstPoint" was the second point in the line before )
        numLines = lastRow - firstRow + 1
        corVal = int(isEnd) * firstRow  # at the end, don't shift the entire matrix up any more -> need to correct for that
        res = 0
        for i in range(0, numLines): res = res + estimatedStimuli_tmp[firstRow+i, lastRow-firstRow-i+corVal]
        if self.sensitivity == 'low': return np.floor(res/numLines)
        if self.sensitivity == 'high': return np.ceil(res/numLines)
        return np.round(res/numLines)
    
    ## =================== MAIN ESTIMATOR FUNCTIONS ====================
    ''' __estimateCoarse: use a sliding test window, in which all possible combinations of low res stimuli are tested '''
    def __estimateCoarse(self, data, aPriori: aPriori, numTestPoints, dt_data):
        if self.verbose: print(f'------------- STEP I (COARSE) --------------- ')
        # get matrix parameters for data, partial model and stimuli
        T_total, T_test, T_vary = self.__getMatrixParams(data, aPriori, numTestPoints, dt_data)
        # get number of points for required matrice
        nT_S_vary__model = np.int32(np.round(T_vary/self.p_model.dt))  # number of points in stimulus__model that vary (-> combo)
        numTestPoints = min(numTestPoints, self.p_S_total.nT)
        numWindows = self.p_S_total.nT - numTestPoints + 1  # number of sliding test windows, can be decreased by number of data points in window
        # init test stimuli
        S_test__model = np.zeros(self.p_model.nT)  # upsampled test stimulus; model.nT ~ T_test
        combinations = self.__getCombinations(numTestPoints)  # get all possible combinations of test points
        for combo in combinations:  # delete those that do not meet aPriori
            if not self.__comboMeetsApriori_inside(combo, aPriori): combo = []
            warn('test if combinations are deleted')
        aPriori__S = aPriori.convert(conversionFactor=1/self.p_S_test.dt, unitName='dt_S')  # convert from [seconds] to [dt_S] -> number of points in stimulus
        S_estimate_tmp = np.zeros([numTestPoints, numTestPoints])  # saves all estimated points for each window -> average them later
        S_estimate_coarse = np.zeros(self.p_S_total.nT)  # result of coarse estimation
        # for each sliding window
        y0 = self.IVP.y_properties.y0  # init for IVP
        for firstTestPoint in range(0, numWindows):
            if self.verbose: print(f'      ... iterating window: firstPoint = {firstTestPoint} ...')
            timeaxis_data, i_start_data, i_end_data = self.__getTimeAxisOfData(firstTestPoint, T_test, T_total)  # get time signatures of all data points that lie within T_test
            if len(timeaxis_data) < 2: 
                numWindows = firstTestPoint
                break
            indice_model = self.__convertDataAxisToModelIndice(timeaxis_data)  # time points within T_test that have signal points close to corresponding data points
            # for all combinations of stimulus points
            RMSE = self.p_model.nT  # init with large number
            for combo in combinations:
                if not self.__comboMeetsApriori(combo, aPriori__S, S_estimate_coarse, firstTestPoint): continue
                if self.verbose: print(f'          ... iterating signal: combo = {combo} ...')
                S_test__model[0:nT_S_vary__model] = upsample(combo, nT_S_vary__model)  # rest of S_test = 0, nT ~ T_test
                signal_estimate_partial, y0_tmp = self.calculateForwardModel(S_test__model, y0=y0)
                RMSE_new = self.__getFitError(signal_estimate_partial[:,1:][:,indice_model], data[:, i_start_data:i_end_data+1])
                if RMSE_new < RMSE:
                    S_estimate_tmp[-1,:] = combo
                    RMSE = RMSE_new
                    y0_next = y0_tmp
            # estimate stimulus for firstTestPoint by averaging all available estimates
            S_estimate_coarse[firstTestPoint] = self.__averagePoint(S_estimate_tmp, firstRow=max(0, numTestPoints-1-firstTestPoint), lastRow=numTestPoints-1)
            S_estimate_tmp[0:numTestPoints-1,:] = S_estimate_tmp[1:numTestPoints,:]  # shift one row up
            y0 = y0_next
            if self.verbose: print(f'              current stimulus estimate: {S_estimate_coarse[0:firstTestPoint+1]}')
        # estimate remaining points (that are never a firstPoint)
        for point in range(numWindows, numWindows+numTestPoints-1):
            S_estimate_coarse[point] = self.__averagePoint(S_estimate_tmp, firstRow=point-numWindows+1, lastRow=numTestPoints-1, isEnd=True)
        return S_estimate_coarse, T_total

    # data should be [D, nT]
    # time frames: __model, __S, __data
    # notation: __model -> matched to time frame of model (etc); if no __x -> native time frame
    def estimateStimulus(self, data, aPriori: aPriori, \
                         coarseTestPoints=3, dt_data=None, return_size='data'):  # return_size in ['model', 'data']
        if self.verbose: print(f'------------- ESTIMATING STIMULUS --------------- ')
        S_estimate_coarse, T_total = self.__estimateCoarse(data, aPriori, coarseTestPoints, dt_data)
        # todo: fine estimation
        if return_size == 'model': S_estimate_coarse = upsample(S_estimate_coarse, T_total/self.p_model.dt)
        if self.verbose: print(f'         FINISHED STIMULUS ESTIMATION.')
        if self.verbose: print(f'         estimated stimulus = {S_estimate_coarse}\n')
        return S_estimate_coarse

    def testEstimator(self, stimulus_hi, aPriori: aPriori, noise_level=0.2, noise_type='additive', nIt=5):
        if self.verbose: print(f'------------- TESTING ESTIMATOR --------------- ')
        if self.verbose: print(f'             S = {stimulus_hi.shape}')
        # get clean data
        dt_data_lo = self.params.boldparams['TR']
        nT_data_lo = np.int32( stimulus_hi.shape[-1] * self.params.dt / dt_data_lo )
        data_clean_hi = self.calculateForwardModel(stimulus_hi)
        data_clean_lo = downsample(data_clean_hi, nT_data_lo)
        # plot original stimulus
        t_end = self.params.nT * self.params.dt
        fig_overview = newfig(2)
        plotManyFunctions(fig_overview[0], stimulus_hi, t1=t_end, label='original', title='stimulus', color='r', linewidth=2)
        plotManyFunctions(fig_overview[1], data_clean_hi[0,:], t1=t_end, label='original', title='signal', color='r', linewidth=2)
        # iterate a few times
        for i in range(0,nIt):
            if self.verbose: print(f'      ITERATION: {i}  ')
            # get noisy data
            if noise_type == 'additive': noise = (np.random.random([self.params.numDepths, nT_data_lo])-0.5) * (np.max(data_clean_lo)-np.min(data_clean_lo)) * noise_level
            data_noisy_lo = data_clean_lo + noise
            estimated_stimulus_hi = self.estimateStimulus(data_noisy_lo, aPriori, dt_data=dt_data_lo, coarseTestPoints=2, return_size='model')
            estimated_data_hi = self.calculateForwardModel(estimated_stimulus_hi)
            estimated_stimulus_hi2 = self.estimateStimulus(data_noisy_lo, aPriori, dt_data=dt_data_lo, coarseTestPoints=2, return_size='model')
            estimated_data_hi2 = self.calculateForwardModel(estimated_stimulus_hi2)
            # plot estimates
            plotManyFunctions(fig_overview[0], estimated_stimulus_hi, t1=t_end, label=f'estimate {i}', linestyle='dashed')
            plotManyFunctions(fig_overview[1], estimated_data_hi[0,:], t1=t_end, label=f'estimate {i}', linestyle='dashed')
            plotDepthwise(data_noisy_lo, self.params.numDepths, mat2D_2=estimated_data_hi, title=f'estimate {i}', legend1='data', legend2='estimate', t1=t_end)
            
            plotManyFunctions(fig_overview[0], estimated_stimulus_hi2, t1=t_end, label=f'estimate2p {i}', linestyle='dashed')
            plotManyFunctions(fig_overview[1], estimated_data_hi2[0,:], t1=t_end, label=f'estimate2p {i}', linestyle='dashed')
            plotDepthwise(data_noisy_lo, self.params.numDepths, mat2D_2=estimated_data_hi2, title=f'estimate2p {i}', legend1='data', legend2='estimate2p', t1=t_end)
        fig_overview[0].legend()
        plt.show()
        if self.verbose: print(f'------------- FINISHED TEST ----------------- ')
    
    # todo: 
    #   calculate (and fit) model longer than stimulus window
    #   add fine-tuning of signal [0 1 0] -> [ 0 0 0 1 1 1 0 0 0] or [0 0 1 1 1 1 0 0 0] etc
    #       search window = [foundPoint[0]-TR:foundPoint[-1]+TR]
    #   fine-tune signal depth-wise
            

''' =============================  EXECUTE ======================== '''
if __name__ == '__main__':
    # parameter files
    parameter_file = "/Havliceks_Implementation2019_example1.txt"
    neural_parameter_file = "/NeuralParameters_210812.txt"
    input_function_file = parameter_file
    params, neural_params, input_TL = readFiles(parameter_file, neural_parameter_file, input_function_file)
    # calculate model
    inverse = hemodynamic_model_inverse(params, neural_params)
    S = input_TL.stimulus[0,0:1500]
    Priori = aPriori(maxDelay=5)
    inverse.testEstimator(S, Priori)

    print()
