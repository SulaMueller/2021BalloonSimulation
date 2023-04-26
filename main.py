
"""
@project:   3D balloon
@name:      main
@author:    Sula Spiegel
@change:    26/04/2023

@reference: [Havlicek2020]: Havlicek M, Uludağ K.: A dynamical model of the laminar BOLD response. NeuroImage. 2020 Jan 1;204:116209.

@project summary:   
    * calculate BOLD and VASO signals from a given binary stimulus
    * use the model & model parameters suggested by Havlicek2020
@background:
    The human cortex is organized as several individual layers perpendicular to the cortical
    surface. Layer specific fMRI aims to localize brain activity on those layers ("depths"). Two methods 
    are available: BOLD and VASO. VASO is highly specific but suffers from poor SNR and low
    availability. BOLD is widely avaible but specificity is hampered by vascular smearing 
    effects: deoxygenated blood is transported to pial veins at the cortical surface, passing 
    through higher layers. Thus, activation in lower layers causes BOLD-signal in higher
    layers by draining veins.
    Hemodynamic modelling aims to reduce vascular smearing thus retrieve the "clean" 
    stimulus function.

model stages:
    Stimulus: describes whether there is an activating stimulus or not
        -> serves as input to the model
    Neural Activation: reaction of neurons to the stimulus
    Balloon Model: reaction of the vessel system to the neural activation
        - mass balanced equations where inflow = outflow + stored blood for each layer
        - outflow of one compartment serves as inflow to the next
        - venous outflow of one layer is additional inflow to veins on the next layer
        - assumes elastic vessel walls -> blood volume storage capacities
        - 3 properties are relevant for BOLD signal: flow, volume, deoxygenation fraction
    BOLD/VASO signal: MRI measurable signal change

@implementation
scalars:
    numDepths: number of cortical layers (default = 6)
    numCompartments: number of vascular compartments to be considered for the model
            default = 3 
                - arteriole: arterial supply (irrelevant for BOLD signal, but provides input to balloon)
                - venule: microvasculature
                - vein: draining veins
    T: number of time points
    nt: time integration step
    ... many model parameters specified in the model classes

time dependent functions:  [dims]; class they are stored in
    Stimulus: [numDepths, T]; Input_Timeline.stimulus
    Neuronal Activiation Function: [numDepths, T]; Neural_Model.n_excitatory
    Inhibitory Neuronal Response: [numDepths, T]; Neural_Model.n_inhibitory
    Vascular Active Signal: [numDepths, T]; Neural_Model.vas
    Arterial Inflow: [numDepths, T]; Input_Timeline.f_arteriole
    Flow: [numCompartments, numDepths, T]; Balloon.flow  (flow[0,:,:]=f_arteriole)
    Vessel Volume Fraction: [numCompartments, numDepths, T]; Balloon.volume
    Fraction of Deoxygenated Blood: [numCompartments, numDepths, T]; Balloon.q
    BOLD signal: [numDepths, T]; BOLD.BOLDsignal
    VASO signal: [numDepths, T]; BOLD.VASOsignal

classes:
    * Model Parameter Classes:
            -> store all parameters required for model calculation
            -> need parameter files <'file.txt'> as input to init
        - Neural_Parameters: parameters for the neural activiation function
        - Model_Parameters: balloon and bold parameters (info about vessel anatomy, scan parameters etc)
    * Input Class: Input_Timeline
            -> input to the model (can be stimulus or f_arteriole)
    * Model Classes:
            -> do the actual model calculations at init
            -> need previous model stage as input
        - Neural_Model
        - Balloon
        - BOLD
    * convenience classes:
        - Plots
            -> easy access to plot dependencies
            -> can have any/all of the models as input
            -> does not do anything at init, only plots if a callable function is called:
                - plotOverAnother -> needs two functions as input
                - plotOverTime -> plot a single function over time
                - plotAll -> plot a list of functions over time
                (example code provided below)
        - Signal_Model 
            -> summarizes all other classes as Signal_Model.neural_params .params .neural .balloon .bold .plots
            -> needs 3 files as input: parameter_file, neural_parameter_file, input_function_file
            -> gets instances of all models at init
"""

import matplotlib.pyplot as plt
import numpy as np

from class_SignalModel import Signal_Model
from writeFile import changeInputFunction, changeMatrixCol, changeMatrixVal, changeValue
from compare2MatLAB import compareMatWithMatfile

# implement 2 example stimuli
appendix = ['example1', 'example2']
for attr in appendix:
    # parameter files
    parameter_file = f"/Havliceks_Implementation2019_{attr}.txt"
    if not '2' in attr: neural_parameter_file = "/NeuralParameters_210812.txt"
    else: neural_parameter_file = "/NeuralParameters_210812 Havliceks_example2.txt"
    input_function_file = parameter_file  # input time course (same file as parameter_file, but does not have to be)
    #changeValue(input_function_file, 'number of time points', new_val=2000)
    #changeMatrixCol(input_function_file, 'type of input', [0,100,300], k=0, numDepths=params.numDepths)
    #changeInputFunction(input_function_file, params.numDepths, new_type='n')

    # create models
    signal = Signal_Model(parameter_file, neural_parameter_file, input_function_file)
    params = signal.params
    neural = signal.neural
    balloon = signal.balloon
    bold = signal.bold
    plots = signal.plots
    
    # plot
    depth = 2
    compartment = params.VENULE
    plots.plotAll(title=attr)  # , depth=depth, compartment=compartment
    #plots.plotOverTime('bold')
    plots.plotOverAnother(balloon.flow[compartment,depth,:], balloon.volume[compartment,depth,:], 'flow', 'volume', \
                          title=f'{attr}, volume-flow-dependency, depth={depth}, {params.COMPARTMENTS[compartment]}')
   
    '''
    # compare to Havliceks implementation in MatLAB
    fdir = 'C:/Users/someone/Desktop/work_stuff/BOLD-VASO model/Matlab-vars/'
    fnames = ['neuro', 'flow_mv', 'volume_mv', 'volume_av', 'dhb_mv', 'dhb_av', 'bold']
    dic = {
        'neuro': neural.n_excitatory,
        'flow_mv': balloon.flow[params.VENULE,:,:],
        'volume_mv': balloon.volume[params.VENULE,:,:],
        'volume_av': balloon.volume[params.VEIN,:,:],
        'dhb_mv': balloon.q[params.VENULE,:,:],
        'dhb_av': balloon.q[params.VEIN,:,:],
        'bold': bold.BOLDsignal
    }
    print('\n============ COMPARISON TO MATLAB VALUES ==================\n')
    for key,value in dic.items():
        compareMatWithMatfile(value, fdir+key+a+'.txt', type = 'double', description=f"{key} ({a})")
    '''

plt.show()
print("Done.")