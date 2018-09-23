
from IFEnsembleDensityIntegratorRWF import IFEnsembleDensityIntegratorRWF

class IFEnsembleDensityIntegratorRWFNI(IFEnsembleDensityIntegratorRWF):

    def __init__(self, nVSteps, leakage, hMu, hSigma, inputCurrentE,
                       nInputsPerNeuron):
        super(IFEnsembleDensityIntegratorRWFNI, self).__init__(nVSteps, 
                                                                leakage, 
                                                                hMu, 
                                                                hSigma, 
                                                                inputCurrentE,
                                                                nInputsPerNeuron)

    def prepareToIntegrate(self, t0, tf, dt, spikeRate0, eInputCurrent):
        super(IFEnsembleDensityIntegratorRWFNI, self).\
         prepareToIntegrate(t0=t0, tf=tf, dt=dt, eInputCurrent=eInputCurrent)
        self._spikeRates[0] = spikeRate0

    def _deriv(self, t, rho):
        spikeRateAtT = self.computeSpikeRate(rho=rho, t=t)
        sigmaE = self._inputCurrentE(t)+self._nInputsPerNeuron*spikeRateAtT
        answer = (self._a0+sigmaE*self._a1).dot(rho)
        return(answer)

    def _computeSpikeRate(self, rho, t):
        nVSteps = rho.size
        dv = 1.0/nVSteps
        dotProduct = self._reversedQs.dot(rho)*dv
        r = self._eInputCurrent(t)*dotProduct/\
            (1-self._nInputsPerNeuron*dotProduct)
        return(r)

