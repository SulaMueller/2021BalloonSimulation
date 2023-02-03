"""
@name:      DependencyGenerator
@author:    Sula Spiegel
@change:    20/01/2023

@summary:   get dependency of overall signal from single variables
"""

from class_main_signalModel import SignalModel


class DependencyGenerator:
    def __init__(self, parent:SignalModel):
        self.parent = parent

    def __iterate(self, varname, minVal, maxVal, index=[], numIt=100, dependentVFT='tau0'):
        stepsize = (maxVal - minVal) / numIt
        for i in range(minVal, maxVal, stepsize):
            self.parent.params.changeVar(varname, i, index, dependentVFT)
            self.parent.createModelInstances()