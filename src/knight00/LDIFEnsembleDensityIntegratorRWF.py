
import math
import numpy as np
from ifEDMsFunctions import computeQs
from LDIFEnsembleDensityIntegratorR import LDIFEnsembleDensityIntegratorR

class LDIFEnsembleDensityIntegratorRWF(LDIFEnsembleDensityIntegratorR):

    def __init__(self, nVSteps, leakage, hMu, hSigma, nInputsPerNeuron, nEigen,
                       eigenRepos):
        super(LDIFEnsembleDensityIntegratorRWF, self).__init__(nVSteps=nVSteps,
                                                              leakage=leakage,
                                                              hMu=hMu, 
                                                              hSigma=hSigma,
                                                              nEigen=nEigen,
                                                              eigenRepos=eigenRepos)
        self._nInputsPerNeuron = nInputsPerNeuron

    def prepareToIntegrate(self, t0, tf, dt, eInputCurrent):
        super(LDIFEnsembleDensityIntegratorRWF, self).\
         prepareToIntegrate(t0=t0, tf=tf, dt=dt,
                                   eInputCurrent=eInputCurrent)
        nTSteps = round((self._tf-self._t0)/self._dt)
        self._eFeedbackCurrentHist = np.zeros(nTSteps+1)
        self._eFeedbackCurrentHist[0] = 0.0

    def getEFeedbackCurrentHist(self):
        return self._eFeedbackCurrentHist

