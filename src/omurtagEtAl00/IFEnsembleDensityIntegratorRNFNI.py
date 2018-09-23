
from IFEnsembleDensityIntegratorRNF import IFEnsembleDensityIntegratorRNF

class IFEnsembleDensityIntegratorRNFNI(IFEnsembleDensityIntegratorRNF):

    def __init__(self, nVSteps, leakage, hMu, hSigma):
        super(IFEnsembleDensityIntegratorRNFNI, self).__init__(nVSteps=nVSteps,
                                                                leakage=leakage,
                                                                hMu=hMu, 
                                                                hSigma=hSigma)

    def _deriv(self, t, rho):
        spikeRateAtT = self._computeSpikeRate(rho=rho, t=t)
        sigmaE = self._eInputCurrent(t)
        binIndex = round((t-self._t0)/self._dt)
        self._spikeRates[binIndex] = spikeRateAtT
        self._eInputCurrentHist[binIndex] = sigmaE
        answer = (self._a0+sigmaE*self._a1).dot(rho)
        return(answer)

