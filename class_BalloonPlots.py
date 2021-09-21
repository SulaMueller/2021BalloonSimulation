"""
@name:      Balloon_Plots
@author:    Sula Spiegel
@change:    10/08/2021

@summary:   Summary class for possible plots
            includes functions to plot specific time lines
"""

import matplotlib.pyplot as plt
import numpy as np
from warnings import warn

class Balloon_Plots:
    def __init__(self, parent):
        self.parent = parent
        self.params = parent.params
        self.time = np.linspace(0, self.params.N, self.params.N)
    
    ''' __getDims: get dims with:
                        * numDepths -> number of subplots 
                        * numCompartments -> number of plots within a subplot 
                   return dims as 3D [1,1,N], [1,nD,N] or [nC,nD,N] '''
    def __getDims(self, data):
        orig_datashape = data.shape
        data = np.squeeze(data)
        if data.ndim == 1:  # 1D data (eg time)
            numCompartments = 1
            numDepths = 1
        elif data.ndim == 2:   # 2D data -> can be plotted in single subplot
            if len(orig_datashape) == 3:  # have a look at original data
                numCompartments = orig_datashape[0]  # assume [numC, numD, N]
                numDepths = orig_datashape[1]
            else:
                numCompartments = 1
                numDepths = data.shape[0]  # assume [numC/D, N]
        elif data.ndim == 3: 
            numCompartments = data.shape[0]  # assume [numC, numD, N]
            numDepths = data.shape[1]
        else:
            warn("ERROR: __getDims can only handle maxDim=3 data.")
            return -1, -1
        return numCompartments, numDepths
    
    ''' __checkDimsMatch: check if x and y have same length '''
    def __checkDimsMatch(self, x, y, xname, yname):
        if x.shape[-1] != y.shape[-1]:
            warn(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> {xname} and {yname} need same length.")
            return False
        if x.ndim > y.ndim:
            warn(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> {xname} needs at most as many dims as {yname}.")
            return False
        return True
    
    ''' __checkShapesMatch check, that number of depths+compartments are either identical or 1 for x,y '''
    def __checkShapesMatch(self, numCompartmentsX, numDepthsX, numCompartmentsY, numDepthsY, xname, yname):
        if (numCompartmentsX != numCompartmentsY and numCompartmentsX != 1) or (numDepthsX != numDepthsY and numDepthsX != 1):
            warn(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> size({xname}) needs to be 1 or same as {yname} on every dim.")
            return False
        return True
    
    ''' __plotWithColour: plot each vessel compartment with fitting colour ''' 
    def __plotWithColour(self, axs, x, y, i, numCompartmentsY):
        colors = ['red', 'cornflowerblue', 'navy']  # arteriole, venule, vein
        if numCompartmentsY == 3:
            axs.plot(x, y, color=colors[i])
        else:
            axs.plot(x, y)

    ''' __plotSubplots: plot a subplot (corresponding to a depth level)
            axs: subplot handle
            sub_x: data on x-axis (1D [N]; could be time)
            sub_y: data on y-axis (1D [N] or 2D [numCompartments, N])
            xname, yname: axis titles '''
    def __plotSubplots(self, axs, sub_x, sub_y, xname, yname):
        # make sure sub_y is 2D
        if sub_y.ndim == 1:
            numCompartmentsY = 1
            sub_y = np.resize(sub_y, (1, sub_y.shape[-1]))
        else: numCompartmentsY = sub_y.shape[0]
        # plot each y-compartment
        for k in range(0, numCompartmentsY):
            self.__plotWithColour(axs, sub_x, sub_y[k,:], k, numCompartmentsY)
        axs.grid(True)
        axs.set_xlabel(xname)
        axs.set_ylabel(yname)

    ''' plotOverAnother: plot two datasets as x(y) (plot compartments together, depths in subplots)
            x: independent dataset as 1D [N], 2D [numC, N] or 3D [numC, numD, N] 
                (if 3D, will have numC*numD subplots)
            y: dependent dataset as 1D [N], 2D [numC, N] or 3D [numC, numD, N] 
                (should have at least as many dims as x) 
            xname, yname: axis titles  '''
    def plotOverAnother(self, x, y, xname, yname, title=None):
        # get dims
        numCompartmentsX, numDepthsX = self.__getDims(x)
        numCompartmentsY, numDepthsY = self.__getDims(y)
        # make sure x and y can be plotted together
        if not self.__checkDimsMatch(x, y, xname, yname) or \
           not self.__checkShapesMatch(\
            numCompartmentsX, numDepthsX, numCompartmentsY, numDepthsY, xname, yname):
            return
        # get shape of plot
        numLines = numDepthsY
        numColumns = numCompartmentsX
        numTimepoints = x.shape[-1]
        # convert x and y into 3D
        x = np.resize(x, (numCompartmentsX, numLines, numTimepoints))
        y = np.resize(y, (numCompartmentsY, numLines, numTimepoints))
        # init figure
        _, axs = plt.subplots(numLines, numColumns)  
        # set title
        if title is not None:
            if hasattr(axs, '__len__'): a = axs[0]
            else: a = axs
            a.set_title(title)
        # for each subplot [numComp, numDepths, N] <-> [numColumns, numLines, N]
        for L in range(0, numLines): 
            for C in range(0, numColumns):
                sub_x = np.squeeze(x[C,L,:])  # -> 1D
                if numColumns == 1: sub_y = np.squeeze(y[:,L,:])  # -> 1D or 2D
                else: sub_y = np.squeeze(y[C,L,:])  # -> 1D
                if numColumns == 1: 
                    if hasattr(axs, '__len__'): ax = axs[L]
                    else: ax = axs
                elif numLines == 1:
                    if hasattr(axs, '__len__'): ax = axs[C]
                    else: ax = axs
                else: ax = axs[L, C]
                self.__plotSubplots(ax, sub_x, sub_y, xname, yname)
    
    ''' __getTimeCourse: extracts specific data defined by name 
            INPUT: varname -> title of plot
            OUTPUT: [data stored in parent by that name, axis title] '''
    def __getTimeCourse(self, varname, depth):
        # define cases
        attrs = ['v', 'f', 'q', 'bold', 'vaso']
        keys = {
            attrs[0]: [['vo', 'Vo', 'VO'], 'volume'],
            attrs[1]: [['f', 'F'], 'flow'],
            attrs[2]: [['q', 'Q', 'ox', 'Ox', 'OX', 'dHb'], 'q'],
            attrs[3]: [['b', 'B'], 'bold'],
            attrs[4]: [['va', 'VA', 'Va'], 'bold']
        }
        # find current case
        for attr in attrs:
            if any([x for x in keys[attr][0] if x in varname]):
                timecourse = getattr(self.parent, keys[attr][1])
                yname = attr
                break
        # if no case found
        if not 'yname' in locals():
            timecourse = getattr(self.parent, varname, -1)
            yname = varname
            if len(timecourse) <= 1:
                warn(f"ERROR: BallonPlots.plotOverTime: unknown variable name {varname}.")
                return 0,''
        # bit more specific for BOLD and VASO
        if yname == attrs[3]:
            timecourse = getattr(timecourse, 'BOLDsignal')
        if yname == attrs[4]:
            timecourse = getattr(timecourse, 'VASOsignal')
        # return specific depth
        if depth > -1: timecourse = timecourse[:, depth, :]
        return np.squeeze(timecourse), yname
    
    ''' plotOverTime: plot data (volume, flow or q) as time line '''
    def plotOverTime(self, varname, depth=-1):
        timecourse, yname = self.__getTimeCourse(varname, depth)
        if len(timecourse) > 1:
            self.plotOverAnother(self.time, timecourse, 't', yname, varname)
    
    ''' plotAll: plot flow, volume and q in one call '''
    def plotAll(self, title='', depth=-1, bold=True, vaso=True):
        if len(title) > 0: comma = ','
        else: comma = ''
        self.plotOverTime(f'flow{comma} {title}', depth)
        self.plotOverTime(f'volume{comma} {title}', depth)
        self.plotOverTime(f'dHb-content{comma} {title}', depth)
        if bold:
            self.plotOverTime(f'BOLD-signal{comma} {title}', depth)
        if vaso:
            self.plotOverTime(f'VASO-signal{comma} {title}', depth)

    
    
        
        

    

