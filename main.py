
import matplotlib.pyplot as plt
import numpy as np
from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters
from class_inputTimeline import Input_Timeline
from class_NeuralModel import Neural_Model
from class_Balloon import Balloon

# parameter files
parameter_file = "/depthDependentBalloonSimulation_210618.txt"
neural_parameter_file = "/NeuralParameters_210812.txt"

# read in parameters
params = Model_Parameters(parameter_file)
neural_params = Neural_Parameters(neural_parameter_file)
input_TimeLine = Input_Timeline(params, parameter_file)

# create neural model
neural_model = Neural_Model(neural_params, params, input_TimeLine)
neural_model.get_activationFunction(plotFlag=True)

# create balloon model
balloon = Balloon(params, input_TimeLine)

# plot 
balloon.plots.plotAll('default')
balloon.plots.plotOverAnother(balloon.flow, balloon.volume, 'flow', 'volume')
balloon.plots.plotOverAnother(balloon.plots.time, balloon.flow[params.VENULE, :,:], 't', 'flow', title='venule')
balloon.plots.plotOverAnother(balloon.plots.time, balloon.flow[params.VEIN, :,:], 't', 'flow', title='vein')


'''
f2 = np.ones([1, input.numDepths, input.N])
f2[0,:, int(input.N/5) : int(input.N*2/5)] = 1.2
balloon.reset_fArteriole(f2)
balloon.plots.plotAll('2')

'''

plt.show()
print("Done.")