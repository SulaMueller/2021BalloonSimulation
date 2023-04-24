
import numpy as np
import matplotlib.pyplot as plt

def loadMatFromTxt(filename, type = 'float'):
    mat = np.loadtxt(filename, delimiter=',', dtype=type)
    return mat

def saveMatToTxt(mat, filename):
    np.savetxt(filename, mat)

def compareMatWithMatfile(mat, matfilename, type = 'float', description = None):
    if description is None: description = matfilename
    mat2 = loadMatFromTxt(matfilename, type)
    if 'bold' in description:
        mat2 = mat2/100  # Havlicek gives result in %
    dif = mat - mat2
    denom1 = np.max(mat) - np.min(mat)
    denom2 = np.max(mat2) - np.min(mat2)
    denom = np.max([denom1, denom2])
    m = np.max(dif)/denom
    if m<0.01:
        print(f"{description} is identical to given matrix.")
    else:
        print(f"Difference between {description} and given matrix: {np.round(m*100, 1)}%")
        compareByPlots(mat, mat2, description)
    return mat2

def compareByPlots(mat1, mat2, description):
    cor_fact = (np.max(mat1)-np.min(mat1))/(np.max(mat2)-np.min(mat2))
    mat2_cor = mat2 * cor_fact
    dif = mat1 - mat2
    # get parameters
    numDepths = np.min(np.shape(mat1))
    N = np.max(np.shape(mat1))
    time = np.linspace(0, N, N)
    # plot functions
    _, axs = plt.subplots(numDepths)
    axs[0].set_title(f"both {description}")
    for D in range(numDepths):
        ax = axs[D]
        ax.plot(time, mat1[D,:])
        ax.plot(time, mat2[D,:])
        ax.grid(True)
    # plot difference
    _, axs = plt.subplots(numDepths)
    axs[0].set_title(f"error of {description}")
    for D in range(numDepths):
        ax = axs[D]
        ax.plot(time, dif[D,:])
        ax.grid(True)
    # print info
    print(f"max1: {np.max(mat1)}, min1: {np.min(mat1)}\nmax2: {np.max(mat2)}, min2: {np.min(mat2)}")
