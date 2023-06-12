"""
@name:      main_signalModel
@author:    Sula Spiegel
@change:    20/04/2023

@summary:   summarize model classes to get one object 
            includes neural, balloon, BOLD/VASO signal model and a Plot-object
"""

import matplotlib.pyplot as plt

from readFile import readFiles
from class_ModelParameters import Model_Parameters
from class_NeuralParameters import Neural_Parameters
from class_inputTimeline import Input_Timeline
from class_NeuralModel import Neural_Model
from class_Balloon import Balloon
from class_BOLD import BOLD
from class_Plots import Plots

class Iterative_Model:
    def __init__(self, \
                 params: Model_Parameters, \
                 neural_params: Neural_Parameters, \
                 input_TL: Input_Timeline ):
        # read in parameters and timeline from files
        self.params, self.neural_params, self.input_TL = params, neural_params, input_TL
        # create model instances
        self.createModelInstances()
    
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
    params, neural_params, input_TL = readFiles(parameter_file, neural_parameter_file, input_function_file)
    model = Iterative_Model(params, neural_params, input_TL)
    model.plots.plotAll()
    plt.show()
    print('end')




