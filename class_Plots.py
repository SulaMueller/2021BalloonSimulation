"""
@name:      Balloon_Plots
@author:    Sula Spiegel
@change:    26/04/2023

@summary:   Summary class for possible plots
            includes functions to plot specific time lines
            at init, doesn't plot, needs extra call to plot
"""

import matplotlib.pyplot as plt
import numpy as np
from warnings import warn
from class_ModelParameters import clearAttrs
from class_NeuralModel import Neural_Model
from class_Balloon import Balloon
from class_BOLD import BOLD

# =========================== CHECKS ==============================
''' getDims: get dims as:
             * numDepths -> number of subplots 
             * numCompartments -> number of colours within a subplot '''
def getDims(data, selfdims=None):
    orig_datashape = data.shape
    data = np.squeeze(data)
    if data.ndim == 1:  # 1D data (eg time)
        numCompartments = 1
        numDepths = 1
    elif data.ndim == 3: 
        numCompartments = data.shape[0]  # assume [numC, numD, T]
        numDepths = data.shape[1]
    elif data.ndim == 2:   # 2D data -> can be plotted in a single subplot
        if len(orig_datashape) == 3:  # have a look at original data
            numCompartments = orig_datashape[0]  # assume [numC, numD, T]
            numDepths = orig_datashape[1]
        elif selfdims is not None:  # use prior info given by Plots.dims
            if selfdims[0]: numCompartments = data.shape[0]
            else: numCompartments = 1
            if selfdims[1] and selfdims[0]: numDepths = data.shape[1]
            elif selfdims[1] and not selfdims[0]: numDepths = data.shape[0]
            else: numDepths = 1
        else:   # assume [numC/D, T]
            if data.shape[0] == 3:  # assume [numC, T], plot in one subplot
                numCompartments = data.shape[0]
                numDepths = 1
            else:  # assume [numD, T], plot each in individual subplot
                numCompartments = 1
                numDepths = data.shape[0]
    else:
        warn("ERROR: __getDims can only handle maxDim=3 data.")
        return -1, -1
    return numCompartments, numDepths

''' checkDimsMatch: check if x and y have same length '''
def checkDimsMatch(x, y, xname, yname):
    if x.shape[-1] != y.shape[-1]:
        warn(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> {xname} and {yname} need same length.")
        return False
    if x.ndim > y.ndim:
        warn(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> {xname} needs at most as many dims as {yname}.")
        return False
    return True

''' checkShapesMatch check, that number of depths+compartments are either identical or 1 for x,y '''
def checkShapesMatch(numCompartmentsX, numDepthsX, numCompartmentsY, numDepthsY, xname, yname):
    if (numCompartmentsX != numCompartmentsY and numCompartmentsX != 1) or (numDepthsX != numDepthsY and numDepthsX != 1):
        warn(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> size({xname}) needs to be 1 or same as {yname} on every dim.")
        return False
    return True

# =========================== PLOT-CLASS ==============================
class Plots:
    def __init__(self, \
                 neural: Neural_Model =None, \
                 balloon: Balloon =None, \
                 bold: BOLD =None ):
        self.neural = neural
        self.balloon = balloon
        self.bold = bold
        for i in [neural, balloon, bold]:
            if i is not None:
                self.params = i.params
                self.time = np.linspace(0, self.params.T, self.params.T)
                break
    
    # =========================== FETCH DATA ==============================
    ''' __getData: extracts specific data defined by name 
            INPUT: varname -> title of plot
            OUTPUT: [data from parents, axis title, plot title] '''
    def __getData(self, varname, depth, compartment, title):
        # define cases
        attrs = ['n', 'f', 'v', 'q', 'bold', 'vaso']  # axis titles
        dic = {  # [[possible varnames], title, obj, obj-attribute, dims: [numC, numD, T]]
            attrs[0]: [['n','N'], 'Neuronal Activation Function (excitation)', self.neural, 'n_excitatory', [False,True,True]],
            attrs[1]: [['f', 'F'], 'Flow Response', self.balloon, 'flow', [True,True,True]],
            attrs[2]: [['vo', 'Vo', 'VO'], 'Volume', self.balloon, 'volume', [True,True,True]],  
            attrs[3]: [['q', 'Q', 'ox', 'Ox', 'OX', 'dHb', 'dhb'], 'dHb-content ("q")', self.balloon, 'q', [True,True,True]],
            attrs[4]: [['b', 'B'], 'BOLD-signal', self.bold, 'BOLDsignal', [False,True,True]],
            attrs[5]: [['va', 'VA', 'Va'], 'VASO-signal', self.bold, 'VASOsignal', [False,True,True]]
        }
        # find current case
        for attr in attrs:
            if any([x for x in dic[attr][0] if x in varname]):
                flag = True
                break
        # if no case found
        if not 'flag' in locals():
            warn(f"ERROR: BallonPlots.plotOverTime: unknown variable name {varname}.")
            return 0,'', ''
        # init title
        if len(title) > 0: comma = ', '
        else: comma = ''
        title = f"{title}{comma}{dic[attr][1]}"
        # get data
        data = getattr(dic[attr][2], dic[attr][3])  # eg. self.balloon.volume
        self.dims = dic[attr][4]
        yname = attr
        # cut data to specific depth and compartment
        numCompartments, numDepths = getDims(data)
        if compartment > -1 and numCompartments > 1: 
            data = np.resize(data, (numCompartments, numDepths, self.params.T))
            data = data[compartment, :, :]
            numCompartments = 1
            self.dims[0] = False
            title = f"{title}, {self.params.COMPARTMENTS[compartment]}"
        if depth > -1 and numDepths > 1:
            data = np.resize(data, (numCompartments, numDepths, self.params.T))
            data = data[:, depth, :]
            self.dims[1] = False
            title = f"{title}, depth={depth}"
        return np.squeeze(data), yname, title

    ''' __retrieveDims: try to find out, which dim of data means what '''
    def __retrieveDims(self, data):  
        if hasattr(self,'dims'): return
        self.dims = [False,False,False]  # [numC, numD, T]
        if np.any(data.shape == self.params.T):               self.dims[2] = True
        if self.params.numCompartments != self.params.numDepths:
            if np.any(data.shape == self.params.numCompartments): self.dims[0] = True
            if np.any(data.shape == self.params.numDepths):       self.dims[1] = True

    # =========================== PLOT-HELPERS ==============================
    ''' __getAx: get specific subplot '''
    def __getAx(self, axs, ind: int =0):
        if hasattr(axs, '__len__'): ax = axs[ind]
        else: ax = axs
        return ax

    ''' __plotWithColour: plot each vessel compartment with fitting colour ''' 
    def __plotWithColour(self, axs, x, y, i, plotAsVesselCompartment):
        if plotAsVesselCompartment:
            if not np.all(y == y[0]):  # only plot, if not constant
                colors = ['red', 'cornflowerblue', 'navy']  # arteriole, venule, vein
                legends = ['arteriole', 'venule', 'vein']
                axs.plot(x, y, color=colors[i], label=legends[i])
        else:
            axs.plot(x, y)

    # =========================== PLOT-CALLABLES ==============================               
    ''' plotOverAnother: plot two datasets as x(y) (plot compartments together, depths in subplots)
            x: independent dataset as 1D [T], 2D [numC, T] or 3D [numC, numD, T] 
                (if 3D, will have numC*numD subplots)
            y: dependent dataset as 1D [T], 2D [numC, T] or 3D [numC, numD, T] 
                (should have at least as many dims as x) 
            xname, yname: axis titles  '''
    def plotOverAnother(self, x, y, xname, yname, title=None):
        # get dims
        self.__retrieveDims(y)  # get self.dims as prior info which dims are used of [numC, numD, T] 
        numCompartmentsX, numDepthsX = getDims(x, self.dims)
        numCompartmentsY, numDepthsY = getDims(y, self.dims)
        # make sure x and y can be plotted together
        if not checkDimsMatch(x, y, xname, yname) or \
           not checkShapesMatch(numCompartmentsX, numDepthsX, numCompartmentsY, numDepthsY, xname, yname):
            clearAttrs(self, ['dims'])
            return
        # get shape of plot
        numLines = numDepthsY
        numTimepoints = x.shape[-1]
        # convert x and y into 3D, repmat if necessary
        x = np.resize(x, (numCompartmentsX, numLines, numTimepoints))
        y = np.resize(y, (numCompartmentsY, numLines, numTimepoints))
        # init figure
        _, axs = plt.subplots(numLines)  
        # set title
        if title is not None: self.__getAx(axs).set_title(title)
        # for each subplot
        for L in range(0, numLines):  # numDepths
            # get panel to plot on
            ax = self.__getAx(axs, L)
            # get title
            if numLines > 1 and self.dims[1]: ytit = f"{yname}(d={(L+1)})"  # plotting depths
            else: ytit = yname
            # for each color
            for C in range(0, numCompartmentsY):
                # get 1D-data for x
                if numCompartmentsX == 1: sub_x = np.squeeze(x[0,L,:])  # always use same 1D x
                else: sub_x = np.squeeze(x[C,L,:])  # if x is not 1D, it should have same number of compartments as y
                # get 1D-data for y
                sub_y = np.squeeze(y[C,L,:])
                # plot
                self.__plotWithColour(ax, sub_x, sub_y, C, numCompartmentsY==self.params.numCompartments)
                ax.grid(True)
                ax.set_xlabel(xname)
                ax.set_ylabel(ytit)
            if L==numLines-1 and numCompartmentsY==self.params.numCompartments: ax.legend()
        clearAttrs(self, ['dims'])
    
    ''' plotOverTime: plot data over time '''
    def plotOverTime(self, varname, depth=-1, compartment=-1, title=''):
        data, yname, title = self.__getData(varname, depth, compartment, title)
        if len(data) > 1: self.plotOverAnother(self.time, data, 't', yname, title=title)
    
    ''' plotAll: plot list of data in one call '''
    def plotAll(self, depth=-1, compartment=-1, title='', lst=['neural', 'flow', 'volume', 'q', 'bold', 'vaso']):
        for key in lst: self.plotOverTime(key, depth, compartment=compartment, title=title)

