
import numpy as np
import matplotlib.pyplot as plt

def newfig(D):
    _, axs = plt.subplots(D)
    return axs

def getTime(mat, t1, t0=0):
    return np.linspace(t0, t1, mat.shape[-1])
    return np.asanyarray(t).reshape(mat.shape[-1])

def plot1D(mat1D, mat1D_2=None, axs=None, title='', legend1='', legend2='', t0=0, t1=None):
    if t1 is None: t1 = len(mat1D)
    t = getTime(mat1D, t1, t0)
    if mat1D_2 is not None: t2 = getTime(mat1D_2, t1, t0)
    if axs is None: _, axs = plt.subplots(1)
    axs.set_title(title)
    axs.plot(t, mat1D[:], label=legend1)
    if mat1D_2 is not None: axs.plot(t2, mat1D_2[:], label=legend2)
    axs.grid('on')
    if len(legend1) > 0 or len(legend2) > 0: axs.legend()

def plotDepthwise(mat2D, D, mat2D_2=None, axs=None, title='', legend1='', legend2='', t0=0, t1=None):
    if t1 is None: t1 = len(mat2D)
    t = getTime(mat2D, t1, t0)
    if mat2D_2 is not None: t2 = getTime(mat2D_2, t1, t0)
    if axs is None: _, axs = plt.subplots(D)
    axs[0].set_title(title)
    for d in range(0, D):
        axs[d].plot(t, mat2D[d,:], label=legend1)
        if mat2D_2 is not None: axs[d].plot(t2, mat2D_2[d,:], '--', label=legend2)
        axs[d].grid('on')
    if len(legend1) > 0 or len(legend2) > 0: axs[0].legend()

def plotManyFunctions(axs, mat1D, t1, t0=0, label='', title=None, *args, **kwargs):
    axs.plot(getTime(mat1D, t1, t0=t0), mat1D, label=label, *args, **kwargs)
    if title is not None: axs.set_title(title)
    axs.grid('on')


