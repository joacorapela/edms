
import pdb
from IFEnsembleDensityIntegratorDDiffusionNF import IFEnsembleDensityIntegratorDDiffusionNF

class IFEnsembleDensityIntegratorDDiffusionNFNI(IFEnsembleDensityIntegratorDDiffusionNF):

    def __init__(self, nVSteps, leakage, hMu):
        super(IFEnsembleDensityIntegratorDDiffusionNFNI, self).\
         __init__(nVSteps=nVSteps, leakage=leakage, hMu=hMu)

    def _deriv(self, t, rho):
        spikeRateAtT = self._computeSpikeRate(rho=rho, t=t)
        sigmaE = self._eInputCurrent(t)
        binIndex = round((t-self._t0)/self._dt)
        self._spikeRates[binIndex] = spikeRateAtT
        self._eInputCurrentHist[binIndex] = sigmaE
        answer = (self._a0+sigmaE*self._a1).dot(rho)
#         pdb.set_trace()
        return(answer)

