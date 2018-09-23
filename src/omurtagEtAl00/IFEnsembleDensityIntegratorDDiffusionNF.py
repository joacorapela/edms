
import pdb
from IFEnsembleDensityIntegratorDDiffusion import IFEnsembleDensityIntegratorDDiffusion

class IFEnsembleDensityIntegratorDDiffusionNF(IFEnsembleDensityIntegratorDDiffusion):

    def _computeSpikeRate(self, rho, t):
        nVSteps = rho.size
        dv = 1.0/nVSteps
        r = self._eInputCurrent(t)*self._hMu**2/dv*max(0.0, rho[-1])
        return(r)

