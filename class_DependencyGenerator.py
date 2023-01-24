"""
@name:      DependencyGenerator
@author:    Sula Spiegel
@change:    20/01/2023

@summary:   get dependency of overall signal from single variables
"""

class DependencyGenerator:
    def __init__(self, parent):
        self.parent = parent
        self.params = parent.params
        self.__get_scalingConstants()
        
    
    ''' __get_scalingConstants: '''
    def __get_scalingConstants(self):
        x = 0