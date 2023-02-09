"""
@name:      main_signalModel
@author:    Sula Spiegel
@change:    20/01/2023

@summary:   summarize model classes to get one object 
            including neural, balloon and BOLD/VASO signal model
"""

from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters
from class_inputTimeline import Input_Timeline
from class_NeuralModel import Neural_Model
from class_Balloon import Balloon

class SignalModel:
    def __init__(self, parameter_file, neural_parameter_file, input_function_file):
        # read in parameters and timeline from files
        self.__readFiles(parameter_file, neural_parameter_file, input_function_file)
        # create model instances
        self.createModelInstances()

    ''' __readFiles: read given parameter files to create parameter dictionaries '''
    def __readFiles(self, parameter_file, neural_parameter_file, input_function_file):
        self.params = Model_Parameters(parameter_file)
        self.neural_params = Neural_Parameters(neural_parameter_file)
        self.input_TimeLine = Input_Timeline(self.params, input_function_file)
        # timeline needs params only for N,numCompartments -> assume those won't be changed
    
    ''' createModelInstances: create instances of model calculation objects '''
    def createModelInstances(self):
        # create neural model (will also set flow in input_TimeLine)
        self.neural_model = Neural_Model(self.neural_params, self.params, self.input_TimeLine)
        # create balloon model (includes BOLD and VASO signals)
        self.balloon = Balloon(self.params, self.input_TimeLine)

''' =============================  EXECUTE ======================== '''
# parameter files
parameter_file = "/depthDependentBalloonSimulation_210618.txt"#"/empty.txt"#
neural_parameter_file = "/NeuralParameters_210812.txt"
input_function_file = parameter_file

signal = SignalModel(parameter_file, neural_parameter_file, input_function_file)


# test
#signal.params.changeVar('F0', 7, [1,3], 's')
#signal.params.changeVar('tau0', 7, [1,3], 's')
#signal.params.changeVar('N', 7)
#print('end')


