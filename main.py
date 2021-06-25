
import matplotlib.pyplot as plt
import numpy as np
from class_Input import Input
from class_Balloon import Balloon

input_file = "/input_depthDependentBalloonSimulation_210618.txt"
input = Input(input_file)
balloon = Balloon(input)
balloon.plots.plotAll('default')
balloon.plots.plotOverAnother(balloon.flow, balloon.volume, 'flow', 'volume')


'''
f2 = np.ones([1, input.numDepths, input.N])
f2[0,:, int(input.N/5) : int(input.N*2/5)] = 1.2
balloon.reset_fArteriole(f2)
balloon.plots.plotAll('2')

'''

plt.show()
print("Done.")