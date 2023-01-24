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
        self.__createModelInstances()

    ''' __readFiles: read given parameter files to create parameter dictionaries '''
    def __readFiles(self, parameter_file, neural_parameter_file, input_function_file):
        self.params = Model_Parameters(parameter_file)
        self.neural_params = Neural_Parameters(neural_parameter_file)
        self.input_TimeLine = Input_Timeline(self.params, input_function_file)
        # timeline needs params only for N,numCompartments -> assume those won't be changed
    
    ''' __createModelInstances: create instances of model calculation objects '''
    def __createModelInstances(self):
        # create neural model (will also set flow in input_TimeLine)
        self.neural_model = Neural_Model(self.neural_params, self.params, self.input_TimeLine)
        # create balloon model (includes BOLD and VASO signals)
        self.balloon = Balloon(self.params, self.input_TimeLine)
    
    ''' __findDictWithParameter: find a specific attribute within given possible dictionaries
        return: the dictionary that has parameterID '''
    def __findDictWithParameter(self, parameterID):
        if parameterID in self.params.__dict__: return self.params
        if parameterID in self.params.boldparams.keys(): return self.params.boldparams
        if parameterID in self.neural_params.__dict__: return self.neural_params

    ''' changeModelParameter: change a single attribute to new_val (find it first) '''
    def changeModelParameter(self, parameterID, new_val):
        x = self.__findDictWithParameter(parameterID)   # find the parameter 
        setattr(x, parameterID, new_val)  # set to new value
        self.__createModelInstances()  # recalculate model instances

    ''' changeModelMatrix: change a matrix attribute 
        a) index=[] -> new_val is entire matrix
        b) index=[x,y...] -> new_val is the new value of matrix at position index '''
    def changeModelMatrix(self, parameterID, new_val, index=[]):
        if index==[]: self.changeModelParameter(parameterID, new_val)
        else:
            x = self.__findDictWithParameter(parameterID)   # find the parameter 
            old_mat = getattr(x,parameterID)
            old_mat[index] = new_val
            self.changeModelParameter(parameterID, old_mat)

''' =============================  EXECUTE ======================== '''
# parameter files
parameter_file = "/depthDependentBalloonSimulation_210618.txt"#"/empty.txt"#
neural_parameter_file = "/NeuralParameters_210812.txt"
input_function_file = parameter_file

signal = SignalModel(parameter_file, neural_parameter_file, input_function_file)

# test
signal.changeModelParameter('B0', 3)
signal.changeModelMatrix('epsilon', [0.2, 0.2, 0.2])
signal.changeModelMatrix('Hct', 2, 2)
            
        

