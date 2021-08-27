
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

''' readValFromText:
DESCRIPTION: extracts value from given file-string by designation of value
INPUT:
    * filetext: content of Parameter_File as string (from getFileText)
    * valuename: name of value that gets extracted (eg 'FILETYPE' or 'TE')
OUTPUT: value of designated valuename in specified type '''
def readValFromText(filetext, valuename, type='float'):
    i = filetext.find(valuename)  # index of first valuename
    i = filetext.find('=', i)   # index of first '=' after valuename
    if i==-1: return None
    substring = filetext[i:len(filetext)]
    value = substring.split()[1]  # returns array of all non-space entries (first should be '=')
    if type=='float': return float(value)
    if type=='int': return int(value)
    if type=='str': return value

''' readMatrixFromText:
DESCRIPTION: extracts matrix from given file-string by designation of matrixname
INPUT:
    * filetext: content of Parameter_File as string (from getFileText)
    * valuename: name of value that gets extracted (eg 'N_P')
    * numCompartments, numDepths:   size(outmatrix) = [numCompartments x numDepths]
    * nVar     : 
            * if several variables are given in the same table (separated by ; or ,): 
                 -> take <nVar>th variable
            * if nVar == -1: take all defined variables (different Vars on dim=2)
    * outmatrix: matrix that is returned (should be empty if given)
OUTPUT: outmatrix [numCompartments, numDepths]; if nVar == -1: [numCompartments, numDepths, numVars] '''
def readMatrixFromText(filetext, valuename, numCompartments, numDepths, nVar=-1, outmatrix=None):
    # get outmatrix
    needNumVars = False
    if outmatrix is None:
        if nVar == -1: needNumVars = True
        else: outmatrix = np.empty([numCompartments, numDepths])

    # find right place in filetext
    k = filetext.find(valuename)  # index of valuename
    if k==-1: return None
    k = filetext.find('|', k)  # index of first '|' after valuename
    substring = filetext[k:len(filetext)]
    substring = substring.replace(',', ';')  # make sure values are separated by ;
    lines = substring.splitlines()  # get array of lines (first line is header)

    # go through all lines 
    for d in range(0, numDepths):
        compartments = lines[d+1].split('|')  # get array with single compartments 
        k_max = min(numCompartments, len(compartments)-2)
        for k in range(0, k_max):
            if not compartments[k+1].strip(): continue  # continue if only white spaces
            values = compartments[k+1].split(';')  # get array of single values within compartment
            if needNumVars:
                needNumVars = False
                numVars = len(values)
                outmatrix = np.empty([numCompartments, numDepths, numVars])

            # write variables into outmatrix
            if nVar > -1:
                outmatrix[k,d] = float(values[nVar].strip())
            else:
                for v in range(0, outmatrix.shape[2]):
                    outmatrix[k,d,v] = float(values[v])
    return outmatrix





