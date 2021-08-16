
"""
@name:      readFile
@author:    Sula Spiegel
@change:    05/08/2021

@summary:   read information from file in specified format
"""

import os
import numpy as np
from warnings import warn

''' getFileText: return total file content as one string '''
def getFileText(filename):
    thispath = os.path.dirname(os.path.realpath(__file__))
    with open(thispath + filename, 'r') as inputfile:
        filetext = inputfile.read()  # total file as string
    return filetext

''' readFloatFromText
DESCRIPTION: extracts value from given file-string by designation of value
INPUT:
    * filetext: content of Parameter_File as string (from getFileText)
    * valuename: name of value that gets extracted (eg 'FILETYPE' or 'TE')
OUTPUT:
    * value of designated valuename as float
    * bool: True if read was successful, False if could not be read (because eg name not found) '''
def readFloatFromText(filetext, valuename):
    i = filetext.find(valuename)  # index of first valuename
    i = filetext.find('=', i)   # index of first '=' after valuename
    if i==-1: return -1, False
    substring = filetext[i:-1]
    value = substring.split()  # returns array of all non-space entries
    return float(value[1]), True  # first entry is '=', second entry should be desired value

''' readMatrixFromText
DESCRIPTION: extracts matrix from given file-string by designation of matrixname
INPUT:
    * filetext: content of Parameter_File as string (from getFileText)
    * valuename: name of value that gets extracted (eg 'N_P')
    * nVar     : 
            * if several variables are given in the same table (separated by ; or ,): 
                 -> take <nVar>th variable
            * if nVar == -1: take all defined variables (different Vars on dim=2)
    * numCompartments, numDepths:   size(outmatrix) = [numCompartments x numDepths]
    * outmatrix: matrix that is returned (should be empty if given)
OUTPUT:
    * outmatrix:  [numCompartments, numDepths]; if nVar == -1: [numCompartments, numDepths, numVars]
    * bool: True if read was successful, False if could not be read (because eg name not found) '''
def readMatrixFromText(filetext, valuename, nVar, numCompartments, numDepths, outmatrix=None):
    # get outmatrix
    if outmatrix is None:
        if nVar == -1:
            warn('ERROR: readMatrixFromText needs nVar or outmatrix as input.')
            return None, False
        outmatrix = np.empty([numCompartments, numDepths])

    # find right place in filetext
    k = filetext.find(valuename)  # index of valuename
    if k==-1: return outmatrix, False
    k = filetext.find('|', k)  # index of first | after valuename
    substring = filetext[k:-1]
    substring = substring.replace(',', ';')  # make sure values are separated by ;
    lines = substring.splitlines()  # get array of lines (first line is header)

    # go through all lines 
    for d in range(0, numDepths):
        compartments = lines[d+1].split('|')  # get array with single compartments 
        k_max = min(numCompartments, len(compartments)-2)
        for k in range(0, k_max):
            if not compartments[k+1].strip(): continue  # continue if only white spaces
            values = compartments[k+1].split(';')  # get array of single values within compartment

            # write variables into outmatrix
            if nVar > -1:
                outmatrix[k,d] = float(values[nVar].strip())
            else:
                for v in range(0, outmatrix.shape[2]):
                    outmatrix[k,d,v] = float(values[v])
    return outmatrix, True





