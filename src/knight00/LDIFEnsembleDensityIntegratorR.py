
import numpy as np
import abc
import pdb
from ifEDMsFunctions import computeQs, computeA1
from LDIFEnsembleDensityIntegrator import LDIFEnsembleDensityIntegrator
from myUtils import getRealPartOfCArrayDotSRIVector

class LDIFEnsembleDensityIntegratorR(LDIFEnsembleDensityIntegrator):

    def __init__(self, nVSteps, leakage, hMu, hSigma, nEigen, eigenRepos):
        super(LDIFEnsembleDensityIntegratorR, self).__init__(nVSteps=nVSteps,
                                                              leakage=leakage,
                                                              nEigen=nEigen,
                                                              eigenRepos=eigenRepos)
        self._a1 = computeA1(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
        qs = computeQs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
        self._reversedQs = qs[-1::-1]

    def _computeSpikeRate(self, sriLDCoefs, rEVecs, sigmaE):
        jImpEAt1 = self._getJImpEAt1(sriLDCoefs=sriLDCoefs, rEVecs=rEVecs, 
                                                            sigmaE=sigmaE)
        return(jImpEAt1)

    def _getJImpEAt1(self, sriLDCoefs, rEVecs, sigmaE):
        nVSteps = self._a0.shape[0]
        dv = 1.0/nVSteps

        bs = rEVecs.transpose().dot(self._reversedQs)
        r = sigmaE*dv*getRealPartOfCArrayDotSRIVector(cArray=bs, 
                                                       sriVector=sriLDCoefs)

        if r<0:
            r = 0.0
        return(r)

    @abc.abstractmethod
    def _getExcitatoryInputCurrent(self, t):
        return

