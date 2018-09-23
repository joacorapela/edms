
import numpy as np
import pdb
import matplotlib.pyplot as plt
from EigenRepos import EigenRepos

class EigenReposForOneStim(EigenRepos):

    def computeEigenDecompositions(self, a0, a1, minStim, maxStim, stepStim,
                                         nEigen):
        self._stimuli = np.arange(start=minStim, stop=maxStim, step=stepStim)
        self._eVals = np.empty((nEigen, self._stimuli.size), dtype=complex)
        self._eVecs = []
        self._dEVals = np.empty((nEigen, self._stimuli.size), dtype=complex)
        self._dEVecs = []

        derDiffMatrix = a1
        prevEVecs = None

        for i in xrange(len(self._stimuli)):
            s = self._stimuli[i]
            print("Processing stimulus %f" % s)
            diffMatrix = a0+s*a1
            eVals, eVecs, aEVecs, sortIndices, flippedEVecs, \
             normalizationIndices = \
             self.computeEigenDecomposition(diffMatrix=diffMatrix, 
                                             prevEVecs=prevEVecs)
#             dEVals, dEScaleMatrix = \
#              self._computeDEvalsAndDEScaleMatrix(dMatrix=derDiffMatrix, 
#                                                           eVals=eVals, 
#                                                           eVecs=eVecs, 
#                                                           aEVecs=aEVecs,
#                                                           normalizationIndices=
#                                                            normalizationIndices)
#             dEVecs = eVecs.dot(dEScaleMatrix)

            prevEVecs = eVecs

            self._eVals[:, i] = eVals[:nEigen]
            self._eVecs.append(eVecs[:, :nEigen])
#             self._dEVals[:, i] = dEVals[:nEigen]
#             self._dEVecs.append(dEVecs[:, :nEigen])

#         pdb.set_trace()

    def getEigenvalues(self, s):
        s0Index = np.argmin(abs(self._stimuli-s))
        s0 = self._stimuli[s0Index]

        eVals0 = self._eVals[:, s0Index]
#         dEVals0 = self._dEVals[:, s0Index]
#         eVals = eVals0 + dEVals0*(s-s0)
        eVals = eVals0

        return(eVals)

    def getEigenvectors(self, s):
        s0Index = np.argmin(abs(self._stimuli-s))
        s0 = self._stimuli[s0Index]

        eVecs0 = self._eVecs[s0Index]
#         dEVecs0 = self._dEVecs[s0Index]
#         eVecs = eVecs0 + dEVecs0*(s-s0)
        eVecs = eVecs0

        return(eVecs)
