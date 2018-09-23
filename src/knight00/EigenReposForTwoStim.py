
import numpy as np
import warnings
import pdb
import matplotlib.pyplot as plt
from EigenRepos import EigenRepos

class EigenReposForTwoStim(EigenRepos):

    def computeEigenDecompositions(self, a0, a1, a2, 
                       startStimE, minStimE, maxStimE, stepStimE,
                       startStimI, minStimI, maxStimI, stepStimI, 
                       nEigen):
        nVSteps = a0.shape[0]
        self._minStimE = minStimE
        self._maxStimE = maxStimE
        self._minStimI = minStimI
        self._maxStimI = maxStimI

        self._stimuliE = np.arange(start=minStimE, stop=maxStimE+stepStimE,
                                                   step=stepStimE)
        iStartStimE = np.argmin(abs(self._stimuliE-startStimE))
        self._stimuliI = np.arange(start=minStimI, stop=maxStimI+stepStimI,
                                                   step=stepStimI)
        iStartStimI = np.argmin(abs(self._stimuliI-startStimI))
        self._eVals = np.empty((nEigen, self._stimuliE.size,
                                         self._stimuliI.size), dtype=complex)
        self._eVecs = np.empty((nVSteps, nEigen, self._stimuliE.size,
                                         self._stimuliI.size), dtype=complex)
#         self._dEValsE = np.empty((nEigen, self._stimuliE.size,
#                                            self._stimuliI.size), dtype=complex)
#         self._dEValsI = np.empty((nEigen, self._stimuliE.size,
#                                            self._stimuliI.size), dtype=complex)
#         self._dEVecsE = np.empty((nVSteps, nEigen, self._stimuliE.size,
#                                            self._stimuliI.size), dtype=complex)
#         self._dEVecsI = np.empty((nVSteps, nEigen, self._stimuliE.size,
#                                            self._stimuliI.size), dtype=complex)

        derDiffMatrixE = a1
        derDiffMatrixI = -a2
        prevEVecs = None

        # Go down along the excitatory stimuli
        for iE in xrange(iStartStimE, -1, -1):
            sE = self._stimuliE[iE]
            # Go down along the inhibitory stimuli
            for iI in xrange(iStartStimI, -1, -1):
                sI = self._stimuliI[iI]
                print("Processing stimuli excitatory=%f and inhibitory=%f" \
                      % (sE, sI))
                eVecs = self._computeAndSaveOneSetOfEigen(a0=a0, a1=a1, a2=a2, 
                                                                 sE=sE, sI=sI,
                                                                 iE=iE, iI=iI,
                                                                 prevEVecs=
                                                                  prevEVecs,
                                                                  nEigen=nEigen)

                if 0<iI and iI<iStartStimI:
                    prevEVecs = eVecs
                elif iI==iStartStimI:
                    prevEVecsAlongI = eVecs
                    prevEVecs = eVecs
                    if iE==iStartStimE:
                        startEVecs = eVecs
                elif iI==0:
                    prevEVecs = prevEVecsAlongI
                else:
                    pdb.set_trace()
                    raise RuntimeError("Non-contemplated case")

            # Go up along the inhibitory stimuli
            for iI in xrange(iStartStimI+1, self._stimuliI.size, 1):
                sI = self._stimuliI[iI]
                print("Processing stimuli excitatory=%f and inhibitory=%f" \
                      % (sE, sI))
                eVecs = self._computeAndSaveOneSetOfEigen(a0=a0, a1=a1, a2=a2, 
                                                                 sE=sE, sI=sI,
                                                                 iE=iE, iI=iI,
                                                                 prevEVecs=
                                                                  prevEVecs,
                                                                  nEigen=nEigen)

                if iI<self._stimuliI.size-1:
                    prevEVecs = eVecs
                else:
                    if 0<iE:
                        prevEVecs = prevEVecsAlongI
                    else:
                        prevEVecs = startEVecs

        # Go up along the excitatory stimuli
        for iE in xrange(iStartStimE+1, self._stimuliE.size, 1):
            sE = self._stimuliE[iE]
            # Go down along the inhibitory stimuli
            for iI in xrange(iStartStimI, -1, -1):
                sI = self._stimuliI[iI]
                print("Processing stimuli excitatory=%f and inhibitory=%f" \
                      % (sE, sI))
                eVecs = self._computeAndSaveOneSetOfEigen(a0=a0, a1=a1, a2=a2, 
                                                                 sE=sE, sI=sI,
                                                                 iE=iE, iI=iI,
                                                                 prevEVecs=
                                                                  prevEVecs,
                                                                  nEigen=nEigen)

                if 0<iI and iI<iStartStimI:
                    prevEVecs = eVecs
                elif iI==iStartStimI:
                    prevEVecsAlongI = eVecs
                    prevEVecs = eVecs
                elif iI==0:
                    prevEVecs = prevEVecsAlongI
                else:
                    raise RuntimeError("Non-contemplated case")

            # Go up along the inhibitory stimuli
            for iI in xrange(iStartStimI+1, self._stimuliI.size, 1):
                sI = self._stimuliI[iI]
                print("Processing stimuli excitatory=%f and inhibitory=%f" \
                      % (sE, sI))
                eVecs = self._computeAndSaveOneSetOfEigen(a0=a0, a1=a1, a2=a2, 
                                                                 sE=sE, sI=sI,
                                                                 iE=iE, iI=iI,
                                                                 prevEVecs=
                                                                  prevEVecs,
                                                                  nEigen=nEigen)

                if iI<self._stimuliI.size-1:
                    prevEVecs = eVecs
                else:
                    prevEVecs = prevEVecsAlongI


    def getEigenvalues(self, sE, sI=0):
        self._checkExtrapolations(sE=sE, sI=sI)

        sE0Index = np.argmin(abs(self._stimuliE-sE))
        sE0 = self._stimuliE[sE0Index]
        sI0Index = np.argmin(abs(self._stimuliI-sI))
        sI0 = self._stimuliI[sI0Index]

        eVals0 = self._eVals[:, sE0Index, sI0Index]
#         dEValsE0 = self._dEValsE[:, sE0Index, sI0Index]
#         dEValsI0 = self._dEValsI[:, sE0Index, sI0Index]
#         eVals = eVals0 + dEValsE0*(sE-sE0) + dEValsI0*(sI-sI0)
        eVals  = eVals0

        return(eVals)

    def getEigenvectors(self, sE, sI=0):
        self._checkExtrapolations(sE=sE, sI=sI)

        sE0Index = np.argmin(abs(self._stimuliE-sE))
        sE0 = self._stimuliE[sE0Index]
        sI0Index = np.argmin(abs(self._stimuliI-sI))
        sI0 = self._stimuliI[sI0Index]

        eVecs0 = self._eVecs[:, :, sE0Index, sI0Index]
#         dEVecsE0 = self._dEVecsE[:, :, sE0Index, sI0Index]
#         dEVecsI0 = self._dEVecsI[:, :, sE0Index, sI0Index]
#         eVecs = eVecs0 + dEVecsE0*(sE-sE0) + dEVecsI0*(sI-sI0)
        eVecs = eVecs0

        return(eVecs)

    def _computeAndSaveOneSetOfEigen(self, a0, a1, a2, sE, sI, iE, iI, 
                                           prevEVecs, nEigen):
        diffMatrix = a0+sE*a1-sI*a2
        derDiffMatrixE = a1
        derDiffMatrixI = -a2

        eVals, eVecs, aEVecs, sortIndices, flippedEVecs, \
         normalizationIndices = \
         self.computeEigenDecomposition(diffMatrix=diffMatrix, 
                                     prevEVecs=prevEVecs)
#         dEValsE, dEScaleMatrixE = \
#         self._computeDEvalsAndDEScaleMatrix(dMatrix=derDiffMatrixE, 
#                                              eVals=eVals, 
#                                              eVecs=eVecs, 
#                                              aEVecs=aEVecs,
#                                              normalizationIndices=
#                                               normalizationIndices)
#         dEVecsE = eVecs.dot(dEScaleMatrixE)
#         dEValsI, dEScaleMatrixI = \
#         self._computeDEvalsAndDEScaleMatrix(dMatrix=derDiffMatrixI, 
#                                              eVals=eVals, 
#                                              eVecs=eVecs, 
#                                              aEVecs=aEVecs,
#                                              normalizationIndices=
#                                               normalizationIndices)
#         dEVecsI = eVecs.dot(dEScaleMatrixI)

        self._eVals[:, iE, iI] = eVals[:nEigen]
        self._eVecs[:, :, iE, iI] = eVecs[:, :nEigen]
#         self._dEValsE[:, iE, iI] = dEValsE[:nEigen]
#         self._dEValsI[:, iE, iI] = dEValsI[:nEigen]
#         self._dEVecsE[:, :, iE, iI] = dEVecsE[:, :nEigen]
#         self._dEVecsI[:, :, iE, iI] = dEVecsI[:, :nEigen]

        return(eVecs)

    def _checkExtrapolations(self, sE, sI):
        if sE>self._maxStimE:
#             pdb.set_trace()
            warnings.warn("Excitatory stimuli %f is larger than the maximum %f" % (sE, self._maxStimE))
        if sE<self._minStimE:
#             pdb.set_trace()
            warnings.warn("Excitatory stimuli %f is smaller than the minimum %f" % (sE, self._minStimE))
        if sI>self._maxStimI:
#             pdb.set_trace()
            warnings.warn("Inhibitory stimuli %f is larger than the maximum %f" % (sI, self._maxStimI))
        if sI<self._minStimI:
#             pdb.set_trace()
            warnings.warn("Inhibitory stimuli %f is smaller than the minimum %f" % (sI, self._minStimI))

