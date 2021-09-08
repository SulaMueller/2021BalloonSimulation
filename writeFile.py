"""
@name:      writeFile
@author:    Sula Spiegel
@change:    27/08/2021

@summary:   change parameters in file automatically
"""

import os
import numpy as np
from warnUsr import warn
from readFile import getFileText, readValFromText, readMatrixFromText

''' __sizeOfSpace: count number of ' ' after startindex until next nonspace character '''
def __sizeOfSpace(string, startindex=0, countChar=' '):
    numS = 0
    for i in range(startindex, len(string)):
        if string[i] is countChar: numS = numS + 1
        else: break
    return numS

''' __writeToFile: save filecontent to file '''
def __writeToFile(filename, filetext):
    thispath = os.path.dirname(os.path.realpath(__file__))
    with open(thispath + filename, 'w') as inputfile:
        inputfile.write(filetext)

''' changeValue: changes specific value in a file
INPUT:
    * filename: path of file (within execution folder)
    * valuename: name of value that gets extracted (eg 'FILETYPE' or 'TE')
    * new_val: value it should have after change
OUTPUT: True if change was successful, False if could not be changed (because eg name not found) '''
def changeValue(filename, valuename, new_val):
    # get line of interest
    filetext_before, line, filetext_after = \
        __findLinesOfInterest(filename, valuename)
    line = line[0]

    # change value
    spacelen = __sizeOfSpace(line, 1)  # number of spaces after '='
    nonspace = line.split()  # get array of all non-space entries on that line
    oldlen = len(nonspace[1])  # get length of original value (first nonspace is '=', second entry should be value)
    line_before = line[0:1+spacelen]  # everything before value
    line_after = line[1+spacelen+oldlen:len(line)]
    newpart = str(new_val)
    filetext = filetext_before + line_before + newpart + line_after + '\n' + filetext_after

    # save file
    __writeToFile(filename, filetext)
    return True  

def deleteValue(filename, valuename):
    changeValue(filename, valuename, '')

''' __findLinesOfInterest: extract lines that are to be changed from a file 
INPUT: * filename: name of original file
       * valuename: name of value to be changed
       * separator: separating valuename from value (can be '=' for single values or '|' for matrice)
       * firstLineAfterName: index of line after valuename in file 
            (0, if on same line; 1, if eg header to consider)
       * numLines: number of lines that will be returned '''
def __findLinesOfInterest(filename, valuename, separator='=', firstlineAfterName=0, numLines=1):
    # get content of file
    filetext = getFileText(filename)

    # find right matrix
    i = filetext.find(valuename)  # index of valuename
    if i==-1: return None, None, None
    i = filetext.find(separator, i)  # index of first =/| after valuename
    
    # find line of interest
    filetext_before = filetext[0:i]
    substring = filetext[i:len(filetext)]
    lines = substring.splitlines()  # get array of lines (matrix: first line is header)
    firstline = firstlineAfterName
    lastline = firstline + numLines - 1
    filetext_before = filetext_before + '\n'.join(lines[0:firstline])
    filetext_after = '\n'.join(lines[lastline+1:len(lines)])
    lines = lines[firstline:lastline+1]
    return filetext_before, lines, filetext_after

''' __changeMatrixVal2: change a value in an already extracted line '''
def __changeMatrixVal2(line, new_vals, k):
    # extract column of interest
    i = -1
    for _ in range(0,k+1): i = line.find('|', i+1)  # index of k_th '|'
    j = line.find('|', i+1)  # index of '|' after column of interest
    if j==-1: j = len(line)
    before_entry = line[0:i+1]
    oldentry = line[int(i+1) : int(j)]
    after_entry = line[j:len(line)]
    # change value
    newpart = ''
    if not hasattr(new_vals, '__len__'): 
        newpart = newpart + str(new_vals)
    else:
        for i in range(0, len(new_vals)):
            newpart = newpart + str(new_vals[i])
            if i < len(new_vals) - 1: newpart = newpart + ';'
            newpart = newpart + ' '
    # adapt spaces
    if len(newpart) < len(oldentry):
        newpart = min(__sizeOfSpace(oldentry), len(oldentry) - len(newpart)) * ' ' + newpart
    if len(newpart) < len(oldentry):
        newpart = newpart + ' ' * (len(oldentry) - len(newpart))
    return before_entry + newpart + after_entry

''' changeMatrixVal: changes specific value in a matrix (in file)
INPUT:
    * filename: path of file (within execution folder)
    * valuename: name of matrix
    * new_vals: array of values a segment should have after change (seperated by ;)
    * k,d: column, line of matrix that will be changed
OUTPUT: True if change was successful, False if could not be changed (because eg name not found) '''
def changeMatrixVal(filename, valuename, new_vals, k, d):
    # find line of interest
    filetext_before, line, filetext_after = __findLinesOfInterest(\
        filename, valuename, separator='|', firstlineAfterName=d+1, numLines=1)
    if filetext_before is None: return False
    # change line
    newline = __changeMatrixVal2(line[0], new_vals, k)
    # save file
    filetext = filetext_before + '\n' + newline + '\n' + filetext_after
    __writeToFile(filename, filetext)
    return True  

def deleteMatrixVal(filename, valuename, k, d):
    changeMatrixVal(filename, valuename, '', k, d)

''' changeMatrixCol: changes entire column of a matrix (in file)
INPUT:
    * filename: path of file (within execution folder)
    * valuename: name of matrix
    * new_vals: array of values a segment should have after change (separated by ;)
    * k: column of matrix that will be changed
    * numDepths: number of columns that will be changed
OUTPUT: True if change was successful, False if could not be changed (because eg name not found) '''
def changeMatrixCol(filename, valuename, new_vals, k, numDepths):
    # find lines of interst
    filetext_before, lines, filetext_after = __findLinesOfInterest(\
        filename, valuename, separator='|', firstlineAfterName=1, numLines=numDepths)
    if filetext_before is None: return False
    # change every line
    for line in lines:
        newline = __changeMatrixVal2(line, new_vals, k)
        filetext_before = filetext_before + '\n' + newline
    # save file
    filetext = filetext_before + '\n' + filetext_after
    __writeToFile(filename, filetext)
    return True  

def deleteMatrixCol(filename, valuename, k, numDepths):
    changeMatrixCol(filename, valuename, '', k, numDepths)

''' changeInputFunction input function can be 'neural' or 'flow'. Change file to new_type. Also adapt values. '''
def changeInputFunction(filename, numDepths, new_type, numSections=None, restval=None, actval=None):
    valuename = 'type of input'
    numCompartments = 2
    filetext = getFileText(filename)
    # check if rewriting is necessary
    old_type = readValFromText(filetext, valuename, 'str')
    same_type = old_type in new_type or new_type in old_type
    if same_type and numSections is None:
        return 
    sections = readMatrixFromText(filetext, valuename, numCompartments, numDepths)
    old_numSections = sections.shape[2]
    if numSections is None: numSections = old_numSections
    if same_type and numSections == old_numSections:
        return
    # get act and rest values depending on input type
    restvals = {'n': 0, 'f': 1}
    actvals = {'n': 1, 'f': 1.2}
    if restval is None: restval = restvals[new_type[0]]
    if actval is None: actval = actvals[new_type[0]]
    new_vals = restval * np.ones([numSections])
    new_vals[1::2] = actval  # set every second element to actval
    # change function in file
    valaxis = 1
    changeValue(filename, valuename, new_type)
    changeMatrixCol(filename, valuename, new_vals, valaxis, numDepths)

