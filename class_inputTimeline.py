
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
from warnUsr import warn
from readFile import getFileText, readValFromText, readMatrixFromText
from class_ModelParameters import Model_Parameters, clearAttrs

class Input_Timeline:
    def __init__(self, params: Model_Parameters, input_file: str, cmro_file=None):
        self.params = params

        self.INPUT_TYPES = ["flow", "neur"]
        self.INDEX_FLOW = 0
        self.INDEX_NEURO = 1
        self.available_input = np.zeros(len(self.INPUT_TYPES)+1)

        self.filetext = getFileText(input_file)  # get file as string
        self.read_input_fromFile()  # read flow_arteriole or neural activation from 
        self.read_cmro(cmro_file)  # read cmro-values, if given
        clearAttrs(self, ['filetext'])

# ---------------------------------  SET INPUT TIMELINE  --------------------------------------
    ''' set_input: assign a given array to matching input structure (flow or neuro) '''
    def __set_fArteriole(self, f_new):
        self.f_arteriole = f_new
    def __set_neural(self, n_new):
        self.neural_input = n_new
    def set_input(self, v_new, inputtype: int):
        if inputtype == self.INDEX_FLOW: self.__set_fArteriole(v_new)
        if inputtype == self.INDEX_NEURO: self.__set_neural(v_new)
        self.available_input[inputtype] = True 

    ''' read_input_fromFile: read f_arteriole or neuro from parameter-file '''
    def read_input_fromFile(self):
        # prepare read-in
        numAxis = 2  # time, value
        timeaxis = 0
        valueaxis = 1
        sections = readMatrixFromText(self.filetext, 'type of input', numAxis, self.params.numDepths)
        numSections = sections.shape[2]

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
        input_type = readValFromText(self.filetext, 'type of input', 'str')
        if not any([x for x in self.INPUT_TYPES if input_type in x or x in input_type]): 
            raise Exception(f'Unrecognized input type. Give any of {self.INPUT_TYPES}.')
        for i in range(0, len(self.INPUT_TYPES)):
            if self.INPUT_TYPES[i] in input_type or input_type in self.INPUT_TYPES[i]: 
                self.set_input(entire_timeline, i)
                break
    
    def read_cmro(self, cmro_file):
        if cmro_file is None: return
        filetext = getFileText(cmro_file)
        lines = filetext.splitlines()  # get array of lines
        self.cmro2 = np.zeros([self.params.numDepths, self.params.N])
        for i in range(0, self.params.N):
            line = lines[i]
            self.cmro2[d,i] = float(line)
            # >> implement here, what needs to be done for each line <<
            raise Exception("\nread_cmro not implemented. Implement according to filestructure if needed.\n")
            # >> then delete Exception <<
        self.available_input[len(self.INPUT_TYPES)] = True 
        
        

    