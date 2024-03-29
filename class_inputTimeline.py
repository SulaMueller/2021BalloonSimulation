
"""
@name:      Input_Timeline
@author:    Sula Spiegel
@change:    25/04/2023

@summary:   Class to store input function (can be stimulus or arterial flow)
@contains:
    * inputtype: stores, which type of time line ("flow" or "stimulus")
    * f_arteriole OR stimulus: [numDepths, numTimepoints] input function
"""

import numpy as np
from warnUsr import warn
from readFile import getFileText, readValFromText, readMatrixFromText
from class_ModelParameters import Model_Parameters, clearAttrs

class Input_Timeline:
    def __init__(self, params: Model_Parameters, input_file: str, cmro_file=None):
        self.params = params

        self.INPUT_TYPES = ["flow", "stimulus"]
        self.INDEX_FLOW = 0
        self.INDEX_STIMULUS = 1
        self.available_input = np.zeros(len(self.INPUT_TYPES)+1)

        self.filetext = getFileText(input_file)  # get file as string
        self.read_input_fromFile()  # read flow_arteriole or stimulus from file
        self.read_cmro(cmro_file)  # read cmro-values, if given
        clearAttrs(self, ['filetext'])

# ---------------------------------  SET INPUT TIMELINE  --------------------------------------
    ''' set_input: assign a given array to matching input structure (flow or stimulus) '''
    def __set_fArteriole(self, f_new):
        self.f_arteriole = f_new
    def __set_stimulus(self, n_new):
        self.stimulus = n_new
    def set_input(self, v_new, inputtype: int):
        if inputtype == self.INDEX_FLOW: self.__set_fArteriole(v_new)
        if inputtype == self.INDEX_STIMULUS: self.__set_stimulus(v_new)
        self.available_input[inputtype] = True 

    ''' read_input_fromFile: read f_arteriole or stimulus from parameter-file '''
    def read_input_fromFile(self):
        # prepare read-in
        numAxis = 2  # time, value
        timeaxis = 0
        valueaxis = 1
        self.sections = readMatrixFromText(self.filetext, 'type of input', numAxis, self.params.numDepths)
        numSections = self.sections.shape[2]

        # read individual sections
        entire_timeline = np.empty([self.params.numDepths, self.params.nT])
        for d in range(0, self.params.numDepths):
            for s in range(0, numSections):
                t0 = int(self.sections[timeaxis, d, s])  # starting time point
                if s < numSections - 1: t1 = int(self.sections[timeaxis, d, s+1])
                else: t1 = self.params.nT  # ending time point
                if t0 < self.params.nT:
                    entire_timeline[d, t0:t1] = self.sections[valueaxis, d, s]
        
        # write timeline to matching structure
        input_type = readValFromText(self.filetext, 'type of input', 'str')
        if not any([x for x in self.INPUT_TYPES if input_type in x or x in input_type]): 
            raise Exception(f'Unrecognized input type. Give any of {self.INPUT_TYPES}.')
        for i in range(0, len(self.INPUT_TYPES)):
            if self.INPUT_TYPES[i] in input_type or input_type in self.INPUT_TYPES[i]: 
                self.set_input(entire_timeline, i)
                self.input_type = self.INPUT_TYPES[i]
                break
    
    def read_cmro(self, cmro_file):
        if cmro_file is None: return
        filetext = getFileText(cmro_file)
        lines = filetext.splitlines()  # get array of lines
        self.cmro2 = np.zeros([self.params.numDepths, self.params.nT])
        for i in range(0, self.params.nT):
            line = lines[i]
            self.cmro2[i] = float(line)
            # >> implement here, what needs to be done for each line <<
            raise Exception("\nread_cmro not implemented. Implement according to filestructure if needed.\n")
            # >> then delete Exception <<
        self.available_input[len(self.INPUT_TYPES)] = True 
        
        

    