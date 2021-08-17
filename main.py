
import matplotlib.pyplot as plt
import numpy as np
from class_ModelParameters import Model_Parameters
from class_Balloon import Balloon


parameter_file = "/depthDependentBalloonSimulation_210618.txt"
params = Model_Parameters(parameter_file)
balloon = Balloon(params)
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