
import matplotlib.pyplot as plt
import numpy as np
'''
from class_SignalModel import Signal_Model
from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters
from class_inputTimeline import Input_Timeline
from class_NeuralModel import Neural_Model
from class_Balloon import Balloon
from writeFile import changeInputFunction, changeMatrixCol, changeMatrixVal, changeValue
from class_DependencyGenerator import DependencyGenerator
from compare2MatLAB import compareMatWithMatfile

# parameter files
parameter_file = "/Havliceks_Implementation2019.txt"#"/empty.txt"#
neural_parameter_file = "/NeuralParameters_210812.txt"
input_function_file = parameter_file
#changeValue(input_function_file, 'number of time points', new_val=2000)
#changeMatrixCol(input_function_file, 'type of input', [0,100,300], k=0, numDepths=params.numDepths)
#changeInputFunction(input_function_file, params.numDepths, new_type='n')


signal = Signal_Model(parameter_file, neural_parameter_file, input_function_file)

neural_model = signal.neural_model
balloon = signal.balloon
params = signal.params

# plot 

neural_model.plot_activationFunction(depth=1)
neural_model.plot_flowResponse(depth=1)
balloon.plots.plotAll()

depth = 2
compartment = params.VENULE
balloon.plots.plotOverAnother(balloon.flow[compartment,depth,:], balloon.volume[compartment,depth,:], 'flow', 'volume')
balloon.plots.plotOverAnother(balloon.plots.time, balloon.flow[params.VENULE, :,:], 't', 'flow', title='venule')
balloon.plots.plotOverAnother(balloon.plots.time, balloon.flow[params.VEIN, :,:], 't', 'flow', title='vein')


#dependency = DependencyGenerator(signal)
#dependency.calculateDependency('B0', 1, 10, numIt=10)

fdir = 'C:/Users/someone/Desktop/work_stuff/BOLD-VASO model/Matlab-vars/'
appendix = '_example1'
fnames = ['neuro', 'flow_mv', 'volume_mv', 'volume_av', 'dhb_mv', 'dhb_av', 'bold']
dic = {
    'neuro': neural_model.n_excitatory,
    'flow_mv': balloon.flow[params.VENULE,:,:],
    'volume_mv': balloon.volume[params.VENULE,:,:],
    'volume_av': balloon.volume[params.VEIN,:,:],
    'dhb_mv': balloon.q[params.VENULE,:,:],
    'dhb_av': balloon.q[params.VEIN,:,:],
    'bold': balloon.bold.BOLDsignal
}

print('\n============ COMPARISON TO MATLAB VALUES ==================\n')
for key,value in dic.items():
    compareMatWithMatfile(value, fdir+key+appendix+'.txt', type = 'double', description=f"{key} ({appendix})")

plt.show()
print("Done.")
'''

import pylops
# import pylops.optimization

test = pylops.optimization.
t = np.arange(63) * 0.004
h, th, hcenter = pylops.utils.wavelets.ricker(t, f0=20)
print('Done.')