


import os
import numpy as np


def getFileText(filename):
    thispath = os.path.dirname(os.path.realpath(__file__))
    with open(thispath + filename,'r') as inputfile:
        inputtext = inputfile.read()  # total file as string
    return inputtext


''' readFloatFromText
DESCRIPTION: extracts value/ input from given file-string by designation of
    value
INPUT:
    * inputtext: content of Input_Parameter_File as string
    * valuename: name of value that gets extracted (eg 'FILETYPE' or 'SDD')
OUTPUT:
    * value of designated valuename as float '''
def readFloatFromText(inputtext, valuename):
    i = inputtext.find(valuename)  # index of valuename
    i = inputtext.find('=', i)   # index of = after valuename
    substring = inputtext[i:-1]
    value = substring.split()  # returns array of all non-space entries
    return float(value[1])  # first entry is '=', second entry should be desired value

''' readMatrixFromText
DESCRIPTION: extracts value/ input from given file-string by designation of
    value
INPUT:
    * inputtext: content of Input_Parameter_File as string
    * valuename: name of value that gets extracted (eg 'N_P')
OUTPUT:
    * value of designated valuename (eg [32,32]) '''
def readMatrixFromText(inputtext, valuename, nVar, numCompartments, numDepths, outmatrix=None):
    # get outmatrix
    if outmatrix is None:
        if nVar == -1:
            print('ERROR: readMatrixFromText needs nVar or outmatrix as input.')
            return None
        outmatrix = np.empty([numCompartments, numDepths])

    # find right place in inputtext
    i = inputtext.find(valuename)  # index of valuename
    i = inputtext.find('|', i)  # index of first | after valuename
    substring = inputtext[i:-1]
    substring.replace(',', ';')  # make sure values are separated by ;
    lines = substring.splitlines()  # get array of lines (first line is header)

    # go through all lines 
    for k in range(0, numDepths):
        compartments = lines[k+1].split('|')  # get array with single compartments 
        for i in range(0, numCompartments):
            if not compartments[i+1].strip(): continue  # continue if only white spaces
            values = compartments[i+1].split(';')  # get array of single values within compartment

            # write variables into outmatrix
            if nVar > -1:
                outmatrix[i,k] = float(values[nVar].strip())
            else:
                for v in range(0, outmatrix.shape[2]):
                    outmatrix[i,k,v] = float(values[v])
    
    return outmatrix





