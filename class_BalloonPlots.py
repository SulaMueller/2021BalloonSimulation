import matplotlib.pyplot as plt
import numpy as np

# summary class for possible plots
class BalloonPlots:
    def __init__(self, parent):
        self.parent = parent
        self.input = parent.input
        self.time = np.linspace(0, self.input.N, self.input.N)
    
    ''' __getDims: get numDepths (-> number of subplots); numCompartments (-> number of plots within a subplot) 
    return data as 3D [1,1,N], [1,nD,N] or [nC,nD,N] '''
    def __getDims(self, data):
        data = np.squeeze(data)
        if data.ndim == 1:  # 1D data (eg time)
            numCompartments = 1
            numDepths = 1
        elif data.ndim == 2:  # 2D data -> can be plotted in single subplot
            numCompartments = data.shape[0]  # assume [numC, N]
            numDepths = 1
        elif data.ndim == 3:  # need several subplots
            numCompartments = data.shape[0]
            numDepths = data.shape[1]  # assume [numC, numD, N]
        else:
            print("ERROR: __getDims can only handle maxDim=3 data.")
            return -1, -1
        return numCompartments, numDepths
        
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
        for i in range(0, numCompartmentsY):
            self.__plotWithColour(axs, sub_x, sub_y[i,:], i, numCompartmentsY)
        axs.grid(True)
        axs.set_xlabel(xname)
        axs.set_ylabel(yname)

    ''' plotOverAnother: plot two datasets as x(y) (plot compartments together, depths in subplots)
    x: independent dataset as 1D [N], 2D [numC, N] or 3D [numC, numD, N] (if 3D, will have numC*numD subplots)
    y: dependent dataset as 1D [N], 2D [numC, N] or 3D [numC, numD, N] (should have at least as many dims as x) '''
    def plotOverAnother(self, x, y, xname, yname, title=None):
        # make sure they have same number of elements
        if x.shape[-1] != y.shape[-1]:
            print(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> {xname} and {yname} need same length.")
            return
        if x.ndim > y.ndim:
            print(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> {xname} needs at most as many dims as {yname}.")
            return
        # get dims
        numCompartmentsX, numDepthsX = self.__getDims(x)
        numCompartmentsY, numDepthsY = self.__getDims(y)
        if (numCompartmentsX != numCompartmentsY and numCompartmentsX != 1) or (numDepthsX != numDepthsY and numDepthsX != 1):
            print(f"SHAPE ERROR: plotOverAnother({xname}({yname})) -> size({xname}) needs to be 1 or same as {yname} on every dim.")
            return
        numLines = numDepthsY
        numColumns = numCompartmentsX
        # make sure x and y are 3D
        x = np.resize(x, (numCompartmentsX, numLines, x.shape[-1]))
        y = np.resize(y, (numCompartmentsY, numLines, x.shape[-1]))
        # init figure
        _, axs = plt.subplots(numLines, numColumns)  
        if title is not None:
            axs[0].set_title(title)
        # for each subplot [numComp, numDepths, N] <-> [numColumns, numLines, N]
        for L in range(0, numLines): 
            for C in range(0, numColumns):
                sub_x = np.squeeze(x[C,L,:])  # -> 1D
                if numColumns == 1: sub_y = np.squeeze(y[:,L,:])  # -> 1D or 2D
                else: sub_y = np.squeeze(y[C,L,:])  # -> 1D
                if numColumns == 1: ax = axs[L]
                else: ax = axs[L, C]
                self.__plotSubplots(ax, sub_x, sub_y, xname, yname)
        
    def __getTimeCourse(self, varname):
        #timecourse = self.parent.exec(varname)
        if varname[0] == 'V' or varname[0] == 'v':
            timecourse = self.parent.volume
            yname = 'v'
        elif varname[0] == 'f' or varname[0] == 'F':
            timecourse = self.parent.flow
            yname = 'f'
        elif varname[0] == 'q' or varname[0] == 'Q' or varname[0] == 'o' or varname[0] == 'O':
            timecourse = self.parent.q
            yname = 'q'
        else:
            print(f"ERROR: BallonPlots.plotOverTime: unknown variable name {varname}.")
            return 0,0
        return np.squeeze(timecourse), yname
    
    def plotOverTime(self, varname): # plot volume, flow or q
        timecourse, yname = self.__getTimeCourse(varname)
        if len(timecourse) > 1:
            self.plotOverAnother(self.time, timecourse, 't', yname, varname)
    
    def plotAll(self, title=''):
        self.plotOverTime(f'flow, {title}')
        self.plotOverTime(f'volume, {title}')
        self.plotOverTime(f'q, {title}')
    
    
        
        

    

