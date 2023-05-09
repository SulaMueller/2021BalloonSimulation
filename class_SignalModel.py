"""
@name:      main_signalModel
@author:    Sula Spiegel
@change:    20/04/2023

@summary:   summarize model classes to get one object 
            includes neural, balloon, BOLD/VASO signal model and a Plot-object
"""

import matplotlib.pyplot as plt

from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters
from class_inputTimeline import Input_Timeline
from class_NeuralModel import Neural_Model
from class_Balloon import Balloon
from class_BOLD import BOLD
from class_Plots import Plots

class Signal_Model:
    def __init__(self, \
                 parameter_file, \
                 neural_parameter_file, \
                 input_function_file ):
        # read in parameters and timeline from files
        self.__readFiles(parameter_file, neural_parameter_file, input_function_file)
        # create model instances
        self.createModelInstances()

    ''' __readFiles: read given parameter files to create parameter dictionaries '''
    def __readFiles(self, parameter_file, neural_parameter_file, input_function_file):
        self.neural_params = Neural_Parameters(neural_parameter_file)
        self.params = Model_Parameters(parameter_file)
        self.input_TL = Input_Timeline(self.params, input_function_file)
        # timeline needs params only for T,numCompartments -> assume those won't be changed
    
    ''' createModelInstances: create instances of model calculation objects 
            (can also be used to re-initialize models) '''
    def createModelInstances(self):
        self.neural = Neural_Model(self.neural_params, self.params, self.input_TL)
        # neural model sets flow in self.input_TL -> input for balloon
        self.balloon = Balloon(self.params, self.input_TL)
        self.bold = BOLD(self.balloon)
        self.plots = Plots(neural=self.neural, balloon=self.balloon, bold=self.bold)


''' =============================  EXECUTE ======================== '''
if __name__ == '__main__':
    # parameter files
    parameter_file = "/Havliceks_Implementation2019_example1.txt"
    neural_parameter_file = "/NeuralParameters_210812.txt"
    input_function_file = parameter_file

    # test
    signal = Signal_Model(parameter_file, neural_parameter_file, input_function_file)
    signal.plots.plotAll()
    plt.show()
    print('end')




