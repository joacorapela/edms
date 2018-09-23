
import sys
import pdb
import numpy as np
from EigenReposForOneStim import EigenReposForOneStim

def main(argv):

    p = 2.0

    aMatrix = np.array([[0, complex(0, p*(-1+p**2)/(1+p**2))], 
                       [complex(0, p*(1+p**2)/(-1+p**2)), 0]])

    dAMatrix = \
     np.array([[0, complex(0, (-1+4*p**2+np.power(p, 4))/(1+p**2)**2)],
               [complex(0, (-1-4*p**2+np.power(p, 4))/(-1+p**2)**2), 0]])

    eigenRepos = EigenReposForOneStim()
    
    eVals, eVecs, aEVecs, _, normalizationIndices = \
     eigenRepos.computeEigenDecomposition(diffMatrix=aMatrix)
    dEVals, dEScalMatrix = \
     eigenRepos._computeDEvalsAndDEScaleMatrix(dMatrix=dAMatrix, 
                                                eVals=eVals, 
                                                eVecs=eVecs, 
                                                aEVecs=aEVecs,
                                                normalizationIndices=
                                                 normalizationIndices)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
