
import numpy as np
import matplotlib.pyplot as plt

from warnUsr import warn
from readFile import readFiles, readFile
from plotHDM import newfig, getTime, plot1D, plotDepthwise, plotManyFunctions
from ivp import hemodynamic_model_forward
from inverse import aPriori, hemodynamic_model_inverse, downsample


class correct_HDR:
    def __init__(self, \
                 params = None, \
                 neural_params = None, \
                 parameter_file = None, \
                 neural_parameter_file = None):
        params, error = self.__getParams(params, parameter_file, 0)
        neural_params, error = self.__getParams(neural_params, neural_parameter_file, 1)
        if error: raise ValueError('Could not create correct_HDR-object. Give params and neural_params as object or file.')
        self.IVP = hemodynamic_model_forward(params, neural_params)
        self.inverse = hemodynamic_model_inverse(params, neural_params)
        self.aPriori = aPriori(maxDelay=5)

    def __getParams(self, p, p_file, case):
        filedescriptions = ['parameter_file', 'neural_parameter_file']
        objectdescriptions = ['Model_Parameters', 'Neural_Parameters']
        error = False
        if p is None:
            if p_file is None: 
                if case<2: warn(f'correct_HDR: give {filedescriptions[case]} or {objectdescriptions[case]}-object.')
                error = True
            else: 
                if case==0: p = readFile(parameter_file=p_file)
                if case==1: p = readFile(neural_parameter_file=p_file)
        return p, error
    
    def __match2dataframe(self, signal__modelframe, dt_data, numDataPoints):
        dt_model = self.IVP.params.dt
        t0 = dt_data/2  # data time points are defined at the middle of the slice
        indice_model = np.zeros(numDataPoints, int)  # can't just slice, since step could be irregular (dt_data is not an exact multiple of dt_model)
        for i in range(0, numDataPoints): indice_model[i] = np.int32(np.round((i*dt_data + t0)/dt_model))
        return signal__modelframe[:,1:][:,indice_model]

    def decontaminate(self, data, dt_data, denoise=True):
        stimulus_estimated = self.inverse.estimateStimulus(data, self.aPriori, dt_data=dt_data)
        signal_decontaminated = self.IVP.solveIVP(stimulus_estimated, modelType='clean')
        numDataPoints = data.shape[-1]
        signal_decontaminated = self.__match2dataframe(signal_decontaminated, dt_data, numDataPoints)
        if not denoise:
            signal_contaminated = self.IVP.solveIVP(stimulus_estimated, modelType='contaminated')
            signal_contaminated = self.__match2dataframe(signal_contaminated, dt_data, numDataPoints)
            noise = data - signal_contaminated
            signal_decontaminated = signal_decontaminated + noise
            return signal_decontaminated, noise
        return signal_decontaminated
    

''' =============================  EXECUTE ======================== '''
if __name__ == '__main__':
    # parameter objects
    parameter_file = "/Havliceks_Implementation2019_example1.txt"
    neural_parameter_file = "/NeuralParameters_210812.txt"
    input_function_file = parameter_file
    params, neural_params, input_TL = readFiles(parameter_file, neural_parameter_file, input_function_file)
    correct_obj = correct_HDR(params=params, neural_params=neural_params)
    # get data
    stimulus_original = input_TL.stimulus
    dt_data_lo = params.boldparams['TR']
    nT_data_lo = np.int32(stimulus_original.shape[-1] * params.dt / dt_data_lo)
    data_clean_hi = correct_obj.IVP.solveIVP(stimulus_original)
    data_clean_lo = downsample(data_clean_hi, nT_data_lo)
    noise_level = 0.2
    noise_original = (np.random.random([params.numDepths, nT_data_lo])-0.5) * (np.max(data_clean_lo)-np.min(data_clean_lo)) * noise_level
    data_noisy_lo = data_clean_lo + noise_original
    # solve
    signal_decontaminated = correct_obj.decontaminate(data_noisy_lo, dt_data_lo)
    plotDepthwise(data_noisy_lo, params.numDepths, mat2D_2=signal_decontaminated, title='decontamination', legend1='original', legend2='decontaminated')
    plt.show()

    print('Done.')