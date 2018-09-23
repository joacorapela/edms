
from IFEnsembleDensityIntegratorR import IFEnsembleDensityIntegratorR
from ifEDMsFunctions import computeQRs

class IFEnsembleDensityIntegratorRNF(IFEnsembleDensityIntegratorR):

    def __init__(self, nVSteps, leakage, hMu, hSigma):
        super(IFEnsembleDensityIntegratorRNF, self).__init__(nVSteps=nVSteps, 
                                                              leakage=leakage,
                                                              hMu=hMu)
        self._reversedQs = computeQRs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)

    def _computeSpikeRate(self, rho, t):
        nVSteps = rho.size
        dv = 1.0/nVSteps
        dotProduct = self._reversedQs.dot(rho)*dv
        r = self._eInputCurrent(t)*dotProduct
        return(r)

