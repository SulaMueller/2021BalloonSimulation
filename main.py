
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

from readFile import readFiles
from class_IterativeModel import Iterative_Model
from Operators import Ops_hemodynamicBOLDmodel
from ivp import hemodynamic_model_forward
from writeFile import changeInputFunction, changeMatrixCol, changeMatrixVal, changeValue

def getMaxDif(mat1, mat2):
    dif = mat1 - mat2
    maxdif = np.max(np.abs(dif)) 
    r = np.max([np.max(np.abs(mat1)), np.max(np.abs(mat2))])
    if r > 0: maxdif = maxdif / r
    if maxdif < 0.000001: maxdif = 0
    return maxdif

def plotSpecificWeirdo(itweird, ivpweird, lineIndex, shift, shiftdir):
    itweird = itweird[lineIndex,:].reshape([1,itweird.shape[1]])
    ivpweird = ivpweird[lineIndex,:].reshape([1,ivpweird.shape[1]])
    plotWeirdos(itweird, ivpweird, shift, shiftdir)

def plotWeirdos(itweird, ivpweird, shift, shiftdir):
    if len(itweird.shape)==1: nLines, nPoints = 1, itweird.shape[0]
    else: nLines, nPoints = np.int32(np.ceil(itweird.shape[0]/2)), itweird.shape[1]
    if shiftdir == 'it':
        itweird = itweird[:,shift:nPoints]
        ivpweird = ivpweird[:,0:nPoints-shift]
    else:
        ivpweird = ivpweird[:,shift:nPoints]
        itweird = itweird[:,0:nPoints-shift]
    _, axs = plt.subplots(nLines)
    for d in range(0, nLines):
        if nLines==1: ax = axs
        else: ax = axs[d]
        ax.plot(itweird[d,:], label='it')
        ax.plot(ivpweird[d,:], label='IVP')
        ax.grid(True)
        if d==0: 
            ax.set_title('venule')
            ax.legend()
    if len(itweird.shape)>1:
        if itweird.shape[0] > 1:
            _, axs = plt.subplots(nLines)
            for d in range(0, nLines):
                if nLines==1: ax = axs
                else: ax = axs[d]
                ax.plot(itweird[d+nLines,:], label='it')
                ax.plot(ivpweird[d+nLines,:], label='IVP')
                ax.grid(True)
                if d==0: 
                    ax.set_title('vein')
                    ax.legend()
    plt.show()

def plotDataTogether(itdata, ivpdata, title):
    if len(itdata.shape) == 2: D = itdata.shape[0] 
    else: D = 1
    _, axs = plt.subplots(D)
    for d in range(0, D):
        if D==1: 
            ax = axs
            itdata_d = itdata
            ivpdata_d = ivpdata
        else: 
            ax = axs[d]
            itdata_d = itdata[d,:]
            ivpdata_d = ivpdata[d,:]
        ax.plot(itdata_d, label='it')
        ax.plot(ivpdata_d, label='IVP')
        ax.grid(True)
        if d==0: 
            ax.set_title(title)
            ax.legend()

def plotFunsTogether(itmodel, ivp_obj, ivp_solution, listOfFuns_ivp, listOfFuns_it):
    for i in range(0, len(listOfFuns_ivp)):
        fun_ivp = listOfFuns_ivp[i]
        fun_it = listOfFuns_it[i]
        ivpdata = ivp_obj.y_properties.getVec(ivp_solution, fun_ivp, 'on')
        itdata_tmp = getattr(itmodel.balloon, fun_it)
        if fun_ivp in ivp_obj.y_properties.compartmented: itdata = itdata_tmp[1,:,:]
        else: itdata = itdata_tmp
        _, axs = plt.subplots(ivp_obj.params.numDepths)
        for d in range(0, ivp_obj.params.numDepths):
            axs[d].plot(itdata[d,:], label='it')
            axs[d].plot(ivpdata[d,:], label='ivp')
            axs[d].grid(True)
        if fun_ivp in ivp_obj.y_properties.compartmented: axs[0].set_title(f'{fun_ivp}_venule')
        axs[0].legend()
        if fun_ivp in ivp_obj.y_properties.compartmented: 
            _, axs = plt.subplots(ivp_obj.params.numDepths)
            itdata = itdata_tmp[2,:,:]
            for d in range(0, ivp_obj.params.numDepths):
                axs[d].plot(itdata[d,:], label='it')
                axs[d].plot(ivpdata[d+ivp_obj.params.numDepths,:], label='ivp')
                axs[d].grid(True)
        axs[0].set_title(f'{fun_ivp}_vein')
        axs[0].legend()


# implement 2 example stimuli
appendix = ['example1']#, 'example2']
for attr in appendix:
    # parameter files
    parameter_file = f"/Havliceks_Implementation2019_{attr}.txt"
    if not '2' in attr: neural_parameter_file = "/NeuralParameters_210812.txt"
    else: neural_parameter_file = "/NeuralParameters_210812 Havliceks_example2.txt"
    input_function_file = parameter_file  # input time course (same file as parameter_file, but does not have to be)
    #changeValue(input_function_file, 'number of time points', new_val=2000)
    #changeMatrixCol(input_function_file, 'type of input', [0,100,300], k=0, numDepths=params.numDepths)
    #changeInputFunction(input_function_file, params.numDepths, new_type='n')

    # read in parameters
    params, neural_params, input_TL = readFiles(parameter_file, neural_parameter_file, input_function_file)

    # create models
    itModel = Iterative_Model(params, neural_params, input_TL)
    plots = itModel.plots

    itdata_dict = dict()
    itdata_dict['n_excitation'] = itModel.neural.n_excitatory
    itdata_dict['f_arteriole'] = itModel.neural.f_arteriole
    itdata_dict['f'] = itModel.balloon.flow[1:3,:,:]
    itdata_dict['v'] = itModel.balloon.volume[1:3,:,:]
    itdata_dict['q'] = itModel.balloon.q[1:3,:,:]
    itdata_dict['BOLD'] = itModel.bold.BOLDsignal

    IVP = hemodynamic_model_forward(params, neural_params)
    t, y, signal = IVP.solveIVP(input_TL.stimulus)

    # plot standard funs
    plotFunsTogether(itModel, IVP, y, ['q'], ['q'])

    # compare weirdos
    shift = 2
    shiftdir = 'it'
    #plotWeirdos(itweird, ivpweird, shift, shiftdir)

    # compare consts
    C_it = np.zeros([3,2,6])
    C_ivp = np.zeros([3,2,6])

    for k in range(0,2):
        for d in range(0,6):
            v = params.V0[k+1,d] * params.numDepths / 100
            C_it[0,k,d] = itModel.bold.consts["c"][0,k+1] * v * (1 - itModel.bold.consts["sV0"][d]) * itModel.bold.consts["H0"][d]
            C_it[1,k,d] = itModel.bold.consts["c"][1,k+1] * v * itModel.bold.consts["H0"][d]
            C_it[2,k,d] = itModel.bold.consts["c"][2,k+1] * v * itModel.bold.consts["H0"][d]
    C_ivp[0,0,:] = IVP.consts.consts['BOLD']['q'][0:6]
    C_ivp[0,1,:] = IVP.consts.consts['BOLD']['q'][6:12]
    C_ivp[1,0,:] = IVP.consts.consts['BOLD']['qv'][0:6]
    C_ivp[1,1,:] = IVP.consts.consts['BOLD']['qv'][6:12]
    C_ivp[2,0,:] = IVP.consts.consts['BOLD']['v'][0:6]
    C_ivp[2,1,:] = IVP.consts.consts['BOLD']['v'][6:12]
    print(f'max difference between constants: {np.max(np.abs(C_it - C_ivp))}')

    b_ivp = IVP.y_properties.getVec(y, 'BOLD', time='on')[:,1:3001]
    b_it = itdata_dict['BOLD'] 
    b_it_scaled = b_it / np.max(b_it) * np.max(b_ivp)
    #plotDataTogether(b_it_scaled, b_ivp, 'corrected BOLD')

    testfuns = ['n_excitation', 'f_arteriole', 'f', 'v', 'q', 'BOLD']
    print('')
    for fun in testfuns:
        ivpdata = IVP.y_properties.getVec(y, fun, time='on')
        # get scaled difference 
        if fun in IVP.y_properties.compartmented: 
            maxdif_venule = np.round(100*getMaxDif(ivpdata[0:6,0:3000], itdata_dict[fun][0,:,:]),2)
            maxdif_vein = np.round(100*getMaxDif(ivpdata[6:12,0:3000], itdata_dict[fun][1,:,:]),2)
            print(f'{fun}_venule: max_dif/max = {maxdif_venule}%')
            print(f'{fun}_vein: max_dif/max = {maxdif_vein}%')
            if maxdif_venule==0 and maxdif_vein==0: continue
        else: 
            maxdif = np.round(100*getMaxDif(ivpdata[0:6,0:3000], itdata_dict[fun]),2)
            print(f'{fun}: max_dif/max = {maxdif}%')
            if maxdif == 0: continue
        # plot both
        _, axs = plt.subplots(IVP.params.numDepths)
        for d in range(0, IVP.params.numDepths):
            if fun in IVP.y_properties.compartmented:
                itdata = itdata_dict[fun][0,d,:]
            else: itdata = itdata_dict[fun][d,:]
            axs[d].plot(itdata, label='iterative')
            axs[d].plot(ivpdata[d,:], label='IVP')
            axs[d].grid(True)
        axs[0].set_title(fun)
        axs[0].legend()
        if fun in IVP.y_properties.compartmented:
            _, axs = plt.subplots(IVP.params.numDepths)
            for d in range(0, IVP.params.numDepths):
                axs[d].plot(itdata_dict[fun][1,d,:], label='iterative')
                axs[d].plot(ivpdata[d+IVP.params.numDepths,:], label='IVP')
                axs[d].grid(True)
            axs[0].set_title(f'{fun}, vein')
            axs[0].legend()
    print('')  
    '''
    plots.plotDataOverTime(signal, 's', 'IVP')
    plots.plotOverTime('BOLD',  title='iterative model')
    
    # plot
    depth = 2
    compartment = params.VENULE
    plots.plotAll(title=attr)  # , depth=depth, compartment=compartment
    plots.plotOverAnother(itModel.balloon.flow[compartment,depth,:], itModel.balloon.volume[compartment,depth,:], 'flow', 'volume', \
                          title=f'{attr}, volume-flow-dependency, depth={depth}, {params.COMPARTMENTS[compartment]}')  '''

plt.show()
print("Done.")