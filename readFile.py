
"""
@name:      readFile
@author:    Sula Spiegel
@change:    05/08/2021

@summary:   read information from file in specified format
"""

import os
import numpy as np
from warnUsr import warn

SEPARATOR_COL = '|'
SEPARATOR_VAL = ';'
ASSIGNATOR = '='

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
    i = filetext.find(ASSIGNATOR, i)   # index of first '=' after valuename
    if i==-1: return None
    substring = filetext[i:len(filetext)]
    value = substring.split()[1]  # returns array of all non-space entries (first should be '=')
    if type=='float': return float(value)
    if type=='int': return int(value)
    if type=='str': return value

''' __getNumCompartments: find out, how many columns a matrix has (excluding first column) '''
def __getNumCompartments(line):
    compartments = line.split(SEPARATOR_COL)  # get array with single compartments 
    return len(compartments) - 2

''' __getNumDepths: find out, how many lines a matrix has (excluding header) '''
def __getNumDepths(lines):
    numDepths = 0
    while lines[numDepths].find(SEPARATOR_COL) > -1: numDepths = numDepths + 1
    return numDepths - 1

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
OUTPUT: outmatrix [numCompartments, numDepths]; if nVar == -1: [numCompartments, numDepths, numVars] 
        (or None, if matrix not readable) '''
def readMatrixFromText(filetext, valuename, numCompartments=-1, numDepths=-1, nVar=-1, outmatrix=None, needCaptions=False):
    # find right place in filetext
    k = filetext.find(valuename)  # index of valuename
    if k==-1: 
        if not needCaptions: return outmatrix
        else: return outmatrix, None, None
    k = filetext.find(SEPARATOR_COL, k)  # index of first '|' after valuename
    substring = filetext[k:len(filetext)]
    substring = substring.replace(',', SEPARATOR_VAL)  # make sure values are separated by ;
    lines = substring.splitlines()  # get array of lines (first line is most of header)
    # find out which parameters are still needed
    if numCompartments == -1: numCompartments = __getNumCompartments(lines[1])
    if numDepths == -1: numDepths = __getNumDepths(lines)
    needNumVars = False
    if outmatrix is None:
        if nVar == -1: needNumVars = True
        else: outmatrix = np.empty([numCompartments, numDepths])
    # init captions
    if needCaptions: 
        header = []
        colcaps = []
    # go through all lines 
    for d in range(0, numDepths+1):
        if not needCaptions and d==0: continue
        compartments = lines[d].split(SEPARATOR_COL)  # get array with single compartments 
        k_max = min(numCompartments, len(compartments)-2)
        for k in range(0, k_max+1):
            if not needCaptions and k==0: continue
            # continue if only white spaces
            if not compartments[k].strip():  
                if needCaptions: 
                    if d==0: header = header + ['']
                    elif k==0: colcaps = colcaps + ['']
                continue
            # write matrix
            if not needCaptions or (k>0 and d>0):
                values = compartments[k].split(SEPARATOR_VAL)  # get array of single values within compartment
                if needNumVars:
                    needNumVars = False
                    numVars = len(values)
                    outmatrix = np.empty([numCompartments, numDepths, numVars])
                # write variables into outmatrix
                if nVar > -1:
                    outmatrix[k-1,d-1] = float(values[nVar].strip())
                else:
                    for v in range(0, outmatrix.shape[2]):
                        outmatrix[k-1,d-1,v] = float(values[v])
            else:  # write captions
                if d==0: header = header + [compartments[k].strip()]
                elif k==0: colcaps = colcaps + [compartments[k].strip()]

    if not needCaptions: return outmatrix
    else: return outmatrix, header, colcaps







