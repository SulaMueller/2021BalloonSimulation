
import matplotlib.pyplot as plt
import numpy as np

from class_SignalModel import Signal_Model
from writeFile import changeInputFunction, changeMatrixCol, changeMatrixVal, changeValue
from compare2MatLAB import compareMatWithMatfile

# implement 2 example time courses
appendix = ['_example1', '_example2']
for attr in appendix:
    # parameter files
    parameter_file = f"/Havliceks_Implementation2019{attr}.txt"
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