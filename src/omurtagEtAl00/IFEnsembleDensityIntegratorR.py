
import numpy as np
import abc
from IFEnsembleDensityIntegrator import IFEnsembleDensityIntegrator
from ifEDMsFunctions import computeA1

class IFEnsembleDensityIntegratorR(IFEnsembleDensityIntegrator):

    def __init__(self, nVSteps, leakage, hMu, hSigma):
        super(IFEnsembleDensityIntegratorR, self).__init__(nVSteps=nVSteps, 
                                                            leakage=leakage,
                                                            hMu=hMu)
        self._a1 = computeA1(nVSteps, hMu, hSigma)
        self._hSigma = hSigma

    def getA1(self):
        return(self._a1)
