
"""
@name:      Input_Timeline
@author:    Sula Spiegel
@change:    25/08/2021

@summary:   Class to store input function (can be neural activation pattern or arterial flow)
@contains:
    * inputtype: stores, which type of time line ("flow" or "neur")
    * f_arteriole OR neural: [numDepths, numTimepoints] input function
"""

import numpy as np
from warnings import warn
from readFile import getFileText, readFloatFromText, readStringFromText, readMatrixFromText
from class_ModelParameters import Model_Parameters

class Input_Timeline:
    def __init__(self, params: Model_Parameters, input_file: str):
        self.params = params

        self.INPUT_TYPES = ["flow", "neur"]
        self.INDEX_FLOW = 0
        self.INDEX_NEURO = 1
        self.available_input = np.zeros(len(self.INPUT_TYPES))

        self.filetext = getFileText(input_file)  # get file as string
        self.read_input_fromFile()  # read flow_arteriole or neural activation from file

# ---------------------------------  SET INPUT TIMELINE  --------------------------------------
    ''' set_input: assign a given array to matching input structure (flow or neuro) '''
    def __set_fArteriole(self, f_new):
        self.f_arteriole = f_new
    def __set_neural(self, n_new):
        self.neural_input = n_new
    def set_input(self, v_new, inputtype):
        if inputtype == self.INDEX_FLOW: self.__set_fArteriole(v_new)
        if inputtype == self.INDEX_NEURO: self.__set_neural(v_new)
        self.available_input[inputtype] = True 
    
    ''' __parse_val: read a single value from the parameter file ''' 
    def __parse_val(self, varname):
        return readFloatFromText(self.filetext, varname)

    ''' read_input_fromFile: read f_arteriole or neuro from parameter-file '''
    def read_input_fromFile(self):
        # prepare read-in
        numSections = int(self.__parse_val("number of input sections")[0])
        numAxis = 2  # time, value
        timeaxis = 0
        valueaxis = 1
        sections = np.empty([numAxis, self.params.numDepths, numSections])
        readMatrixFromText(self.filetext, 'number of input sections', \
            -1, numAxis, self.params.numDepths, sections)

        # read individual sections
        entire_timeline = np.empty([self.params.numDepths, self.params.N])
        for d in range(0, self.params.numDepths):
            for s in range(0, numSections):
                t0 = int(sections[timeaxis, d, s])  # starting time point
                if s < numSections - 1: t1 = int(sections[timeaxis, d, s+1])
                else: t1 = self.params.N  # ending time point
                if t0 < self.params.N:
                    entire_timeline[d, t0:t1] = sections[valueaxis, d, s]
        
        # write timeline to matching structure
        input_type = readStringFromText(self.filetext, 'type of input')[0]
        if not any([x for x in self.INPUT_TYPES if input_type in x or x in input_type]): 
            raise Exception(f'Unrecognized input type. Give any of {self.INPUT_TYPES}.')
        for i in range(0, len(self.INPUT_TYPES)):
            if self.INPUT_TYPES[i] in input_type or input_type in self.INPUT_TYPES[i]: 
                self.set_input(entire_timeline, i)
                break
        
        

    