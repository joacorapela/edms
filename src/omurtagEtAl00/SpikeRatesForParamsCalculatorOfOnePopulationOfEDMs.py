
import pdb
import numpy as np
from IFEnsembleDensityIntegratorRWFWI import IFEnsembleDensityIntegratorRWFWI

class SpikeRatesForParamsCalculatorOfOnePopulationOfEDMs:
    def __init__(self, hMu, hSigma, kappaMu, kappaSigma,
                       t0, tf, dt, nVSteps, rho0, 
                       eExternalInput, iExternalInput):
        self._hMu = hMu
        self._hSigma = hSigma
        self._kappaMu = kappaMu
        self._kappaSigma = kappaSigma
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._nVSteps = nVSteps
        self._rho0 = rho0
        self._eExternalInput = eExternalInput
        self._iExternalInput = iExternalInput

    def calculateSpikeRatesForParams(self, params):
        nTSteps = round((self._tf-self._t0)/self._dt)
        spikeRatesForParams = np.empty((len(params), nTSteps+1))
        for i in xrange(len(params)):
            times, spikeRatesForOnseSetParams = \
             self._calculateSpikeRatesForOneSetParams(oneSetParams=params[i])
            spikeRatesForParams[i, :] = spikeRatesForOnseSetParams
        return(times, spikeRatesForParams)

    def _calculateSpikeRatesForOneSetParams(self, oneSetParams):
        leakage = oneSetParams[0]
        nInputsPerNeuron = oneSetParams[1]
        fracExcitatoryNeurons = oneSetParams[2]

        print("Calculating spike rates for leakage=%.02d, nInputsPerNeuron=%d, fracExcitatoryNeurons=%.02f"%(leakage, nInputsPerNeuron, fracExcitatoryNeurons))

        ifEDIntegrator = \
         IFEnsembleDensityIntegratorRWFWI(nVSteps=self._nVSteps,
                                           leakage=leakage, 
                                           hMu=self._hMu, 
                                           hSigma=self._hSigma, 
                                           kappaMu=self._kappaMu, 
                                           kappaSigma=self._kappaSigma, 
                                           fracExcitatoryNeurons=
                                            fracExcitatoryNeurons, 
                                           nInputsPerNeuron=nInputsPerNeuron)
        ifEDIntegrator.prepareToIntegrate(t0=self._t0, tf=self._tf, dt=self._dt,
                                           eExternalInput=self._eExternalInput,
                                           iExternalInput=self._iExternalInput)
        ifEDIntegrator.setInitialValue(rho0=self._rho0)
        times, _, spikeRates, _, = \
         ifEDIntegrator.integrate(dtSaveRhos=float("Inf"))
        return(times, spikeRates)

