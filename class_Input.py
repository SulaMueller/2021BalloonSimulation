
import numpy as np
from readFile import getFileText, readFloatFromText, readMatrixFromText

class Input:
    def __init__(self, input_file):
        self.numCompartments = 3   # arteriole, venule, vein
        self.__parse_inputFile(input_file)
        self.__init_fArteriole()

    def __init_matrices(self):
        self.V0 = np.empty([self.numCompartments, self.numDepths])
        self.F0 = np.empty([self.numCompartments, self.numDepths])
        self.alpha = np.empty([self.numCompartments, self.numDepths])
        self.vet = np.empty([self.numCompartments, self.numDepths, 2])  # visco-elastic time-constant for in-/deflation
    
    def __parse_val(self, varname):
        return readFloatFromText(self.inputtext, varname)
    
    def __parse_matrix(self, varname, nVar, outmatrix):
        return readMatrixFromText(self.inputtext, varname, nVar, self.numCompartments, self.numDepths, outmatrix)

    def __parse_inputFile(self, filename):
        self.inputtext = getFileText(filename)  # gets total file as string
        self.N = int(self.__parse_val("number of time points"))
        self.numDepths = int(self.__parse_val("number of depth levels"))
        self.__init_matrices()
        self.__parse_matrix('V0', 0, self.V0)
        self.__parse_matrix('F0', 1, self.F0)
        self.__parse_matrix('alpha', 0, self.alpha)
        self.__parse_matrix('visco-elastic time constants', -1, self.vet)
        boldparams_tmp = np.empty([self.numCompartments, 4])
        readMatrixFromText(self.inputtext, 'BOLD', 0, self.numCompartments, 4, boldparams_tmp)
        self.E0 = boldparams_tmp[:,0]
        self.B0 = self.__parse_val("B0")
        self.boldparams = {
            'epsilon': boldparams_tmp[:,1],
            'Hct': boldparams_tmp[:,2],
            'r0': boldparams_tmp[:,3],
            'dXi': self.__parse_val("dXi"),
            'TE': self.__parse_val("TE")
        }

    def set_fArteriole(self, f_new):
        self.f_arteriole = f_new

    def __init_fArteriole(self):
        f_arteriole = np.ones([1, self.numDepths, self.N])
        f_arteriole[0,:, int(self.N/5) : int(self.N*3/5)] = 1.2
        self.set_fArteriole(f_arteriole)
    