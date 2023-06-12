
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

# data should have the shape [D,T]
# stimulus should be [T]
''' hemodynamic_model_inverse: class to compute the inverse of the hemodynamic model -> best guess of stimulus that fits given (signal) data
        ATTRIBUTES:
            * modelparams: Model_Parameters object, dt, nT, D match the model (fine time grid)
            * dataparams: Model_Parameters object, dt, nT, D match the data (much coarser time grid)
        FUNCTIONS: '''
class hemodynamic_model_inverse:
    def __init__(self, 
                 params: Model_Parameters, \
                 neural_params: Neural_Parameters):
        self.modelparams, self.dataparams, self.neural_params = params, copy.deepcopy(params), neural_params
        self.dataparams.dt = self.modelparams.boldparams['TR']
        self.IVP = hemodynamic_model_forward(params, neural_params)
        self.setSensitivity('round')
        self.verbose = True
    
    def setSensitivity(self, sensitivity='low'): self.sensitivity = sensitivity  # define, if estimate is floor'd ('low'), ceil'd ('high') or rounded
    def upsample(self, stimulus_lo_res, numModelPoints=None): 
        if numModelPoints is None: numModelPoints = self.modelparams.nT
        numModelPoints = np.int32(numModelPoints)
        if stimulus_lo_res.ndim==1: return np.repeat(stimulus_lo_res, np.int32(numModelPoints/len(stimulus_lo_res)))[0:numModelPoints]
    def downsample(self, stimulus_hi_res, numDataPoints=None):  # take the point from the middle of each slice
        if numDataPoints is None: numDataPoints = self.dataparams.nT
        numDataPoints = np.int32(numDataPoints)
        if stimulus_hi_res.ndim==1: 
            sliceDist = np.int32(len(stimulus_hi_res)/numDataPoints)
            return stimulus_hi_res[np.int32(sliceDist/2)::sliceDist][0:numDataPoints]
        if stimulus_hi_res.ndim==2: 
            sliceDist = np.int32(stimulus_hi_res.shape[1]/numDataPoints)
            return stimulus_hi_res[:,np.int32(sliceDist/2)::sliceDist][:,0:numDataPoints]
    def __adaptParametersToData(self, data):  
        self.dataparams.numDepths, self.modelparams.numDepths = data.shape[0], data.shape[0]
        if self.dataparams.numDepths != self.IVP.params.numDepths: self.IVP = hemodynamic_model_forward(self.modelparams, self.neural_params)
        self.dataparams.nT = data.shape[1]
        self.modelparams.nT = np.int32(data.shape[1] * self.dataparams.dt / self.modelparams.dt)
        self.IVP.params.nT = self.modelparams.nT
    def __defineWindowSize(self):
        T_min = 4  # window should cover about 4s (which is the delay of most of the effect)
        return np.int32(np.round(T_min / self.dataparams.dt))
    def __getCombinations(self, numWindowPoints):  # get all possible combinations of stimulus values in a given window size (stimulus can be 0 or 1)
        numCombinations = np.int32(2**numWindowPoints)
        combinations = np.zeros([numCombinations, numWindowPoints])
        for i in range(0, numCombinations): combinations[i, :] = int2bin(i,numWindowPoints)
        return combinations
    def __getFitError(self, y, data, errorType='rmse'): 
        if errorType=='rmse': return rmse(y, data)
    def __averagePoint(self, estimatedStimuli_tmp, point_index):  # point_index: in total signal
        # estimatedStimuli_tmp = [numWindows, numWindowPoints]; identical points are diagonal ("firstPoint" was the second point in the line before )
        numWindowPoints =  estimatedStimuli_tmp.shape[1]
        lastOccurence = min(point_index, estimatedStimuli_tmp.shape[0]-1)  # line where the point is on 0th position (and is kicked out after)
        firstOccurence = max(0, point_index-numWindowPoints+1)
        numLines = lastOccurence - firstOccurence + 1
        res = 0
        for i in range(firstOccurence, lastOccurence+1): res = res + estimatedStimuli_tmp[i, lastOccurence-i]
        if self.sensitivity == 'low': return np.floor(res/numLines)
        if self.sensitivity == 'high': return np.ceil(res/numLines)
        return np.round(res/numLines)
    def calculateForwardModel(self, stimulus1D_hi_res, y0=None, method='ivp'):
        if method=='ivp': _, y, s = self.IVP.solveIVP(stimulus1D_hi_res, y0)
        if y0 is None: return s
        return s, y[:,-1]
        
    def estimateStimulus(self, data, return_size='model'):  # return_size in ['model', 'data']
        if self.verbose: print(f'------------- ESTIMATING STIMULUS --------------- ')
        self.__adaptParametersToData(data)
        numWindowPoints = min(self.dataparams.nT, self.__defineWindowSize())
        numWindows = self.dataparams.nT - numWindowPoints + 1
        numModelPoints = np.int32(self.modelparams.nT * numWindowPoints / self.dataparams.nT)  # nP in part of model that is calculated
        numDataPoints = numWindowPoints  # nP in part of model that is calculated (need that part of data for error calculation)
        estimatedStimuli_tmp = np.zeros([numWindows, numWindowPoints])  # saves all window points for each estimated window -> average them later
        estimatedStimulus = np.zeros(self.dataparams.nT)  # result of entire estimation
        combinations = self.__getCombinations(numWindowPoints)  # get all possible stimuli
        # for each sliding window
        y0 = self.IVP.y_properties.y0  # init for IVP
        for firstTestPoint in range(0, numWindows):
            if self.verbose: print(f'      ... iterating window: firstPoint = {firstTestPoint} ...')
            RMSE = self.modelparams.nT  # init with large number
            # for all combinations of signal points
            for combo in combinations:
                if self.verbose: print(f'          ... iterating signal: combo = {combo} ...')
                stimulus_hi_res = self.upsample(combo, numModelPoints)
                signal_estimate_hi_res, y0_tmp = self.calculateForwardModel(stimulus_hi_res, y0=y0)
                RMSE_new = self.__getFitError(self.downsample(signal_estimate_hi_res[:,1:], numDataPoints), data[:, firstTestPoint:firstTestPoint+numDataPoints])
                if RMSE_new < RMSE:
                    estimatedStimuli_tmp[firstTestPoint,:] = combo
                    RMSE = RMSE_new
                    y0_next = y0_tmp
            # estimate stimulus for firstTestPoint by averaging all available estimates
            estimatedValue = self.__averagePoint(estimatedStimuli_tmp, firstTestPoint)
            estimatedStimulus[firstTestPoint] = estimatedValue
            y0 = y0_next
            if self.verbose: print(f'              current stimulus estimate: {estimatedStimulus[0:firstTestPoint+1]}')
        # estimate remaining points (that are never a firstPoint)
        for point in range(numWindows, self.dataparams.nT):
            estimatedStimulus[point] = self.__averagePoint(estimatedStimuli_tmp, point)
        if return_size == 'model': estimatedStimulus = self.upsample(estimatedStimulus)
        if self.verbose: print(f'         FINISHED STIMULUS ESTIMATION.')
        if self.verbose: print(f'         estimated stimulus = {estimatedStimulus}\n')
        return estimatedStimulus

    def testEstimator(self, stimulus_hi_res, noise_level=0.2, noise_type='additive', nIt=5):
        if self.verbose: print(f'------------- TESTING ESTIMATOR --------------- ')
        if self.verbose: print(f'             S = [{stimulus_hi_res.shape}]')
        # prepare parameters
        self.modelparams.nT = stimulus_hi_res.shape[0]
        self.dataparams.nT = np.int32(stimulus_hi_res.shape[0] / self.dataparams.dt * self.modelparams.dt)
        # get clean data
        data_clean_hi = self.calculateForwardModel(stimulus_hi_res)
        data_clean_lo = self.downsample(data_clean_hi)
        # plot original stimulus
        t_end = self.modelparams.nT * self.modelparams.dt
        fig_overview = newfig(2)
        plotManyFunctions(fig_overview[0], stimulus_hi_res, t_end, label='original', title='stimulus', color='r', linewidth=2)
        plotManyFunctions(fig_overview[1], data_clean_hi[0,:], t_end, label='original', title='signal', color='r', linewidth=2)
        # init estimates
        estimated_stimuli_lo = np.zeros([nIt, self.dataparams.nT])
        # iterate a few times
        for i in range(0,nIt):
            if self.verbose: print(f'      ITERATION: {i}  ')
            # get noisy data
            if noise_type == 'additive': noise = (np.random.random([self.dataparams.numDepths, self.dataparams.nT])-0.5) * (np.max(data_clean_lo)-np.min(data_clean_lo)) * noise_level
            data_noisy_lo = data_clean_lo + noise
            estimated_stimuli_lo[i,:] = self.estimateStimulus(data_noisy_lo, return_size='data')
            estimated_data_hi = self.calculateForwardModel(self.upsample(estimated_stimuli_lo[i,:]))
            # plot estimates
            plotManyFunctions(fig_overview[0], estimated_stimuli_lo[i,:], t1=t_end, label=f'estimate {i}', linestyle='dashed')
            plotManyFunctions(fig_overview[1], estimated_data_hi[0,:], t1=t_end, label=f'estimate {i}', linestyle='dashed')
            plotDepthwise(data_noisy_lo, self.dataparams.numDepths, mat2D_2=estimated_data_hi, title=f'estimate {i}', legend1='data', legend2='estimate', t1=t_end)
        fig_overview[0].legend()
        plt.show()
        if self.verbose: print(f'------------- FINISHED TEST ----------------- ')
    
    # todo: 
    #   calculate (and fit) model longer than stimulus window
    #   add fine-tuning of signal [0 1 0] -> [ 0 0 0 1 1 1 0 0 0] or [0 0 1 1 1 1 0 0 0] etc
    #       search window = [foundPoint[0]-TR:foundPoint[-1]+TR]
    #   make stimulus model finer than data (eg twice as many points) for increased precision
    #   include prior knowledge about stimulus (eg duration in range..., distance between stimuli etc) -> make an extra class PriorKnowledge
            

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
    inverse.testEstimator(S)

    print()
