
import pdb
import numpy as np

class SpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs:
    def __init__(self, t0, tf, dt, nVSteps, edm1Rho0, edm2Rho0, 
                       edm1ESigma, edm1ISigma,
                       edm2ESigma, edm2ISigma,
                       twoPIFEDIntegrator):
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._nVSteps = nVSteps
        self._edm1Rho0 = edm1Rho0
        self._edm2Rho0 = edm2Rho0
        self._edm1ESigma = edm1ESigma
        self._edm1ISigma = edm1ISigma
        self._edm2ESigma = edm2ESigma
        self._edm2ISigma = edm2ISigma
        self._twoPIFEDIntegrator = twoPIFEDIntegrator

    def calculateSpikeRatesForParams(self, params):
        nTSteps = round((self._tf-self._t0)/self._dt)
        spikeRatesForParams = np.empty((len(params), nTSteps+1, 2))
        for i in xrange(len(params)):
            times, spikeRatesForOnseSetParams = \
             self._calculateSpikeRatesForOneSetParams(oneSetParams=params[i])
            spikeRatesForParams[i, :, :] = spikeRatesForOnseSetParams
        return(times, spikeRatesForParams)

    def _calculateSpikeRatesForOneSetParams(self, oneSetParams):
        print("Calculating spike rates for ")
        print(oneSetParams)

        if oneSetParams[0] is not None:
            pEDM1ESigma = lambda t: self._edm1ESigma(t=t, w=oneSetParams[0])
        else:
            pEDM1ESigma = self._edm1ESigma

        
        if oneSetParams[1] is not None:
            pEDM1ISigma = lambda t: self._edm1ISigma(t=t, w=oneSetParams[1])
        else:
            pEDM1ISigma = self._edm1ISigma

        if oneSetParams[2] is not None:
            pEDM2ESigma = lambda t: self._edm2ESigma(t=t, w=oneSetParams[2])
        else:
            pEDM2ESigma = self._edm2ESigma

        
        if oneSetParams[3] is not None:
            pEDM2ISigma = lambda t: self._edm2ISigma(t=t, w=oneSetParams[3])
        else:
            pEDM2ISigma = self._edm2ISigma

        self._twoPIFEDIntegrator.\
             prepareToIntegrate(t0=self._t0, tf=self._tf, dt=self._dt,
                                             nVSteps=self._nVSteps, 
                                             edm1EInputCurrent=pEDM1ESigma,
                                             edm1IInputCurrent=pEDM1ISigma,
                                             edm2EInputCurrent=pEDM2ESigma,
                                             edm2IInputCurrent=pEDM2ISigma)
        self._twoPIFEDIntegrator.setInitialValue(edm1Rho0=self._edm1Rho0,
                                                 edm2Rho0=self._edm2Rho0)
        edm1Times, _, edm1SpikeRates, edm2Times, _, edm2SpikeRates, _ = \
         self._twoPIFEDIntegrator.integrate(dtSaveRhos=float("Inf"))
        return(edm1Times, np.column_stack((edm1SpikeRates, edm2SpikeRates)))

