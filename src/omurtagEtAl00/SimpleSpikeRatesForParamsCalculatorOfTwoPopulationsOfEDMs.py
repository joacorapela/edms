
import sys
import pdb
import numpy as np

class SimpleSpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs:
    def __init__(self, eInputs, eRho0, iRho0, twoPIFEDIntegrator):
        self._eInputs = eInputs
        self._eRho0 = eRho0
        self._iRho0 = iRho0
        self._twoPIFEDIntegrator = twoPIFEDIntegrator

    def calculateSpikeRatesForParams(self, params):
        spikeRatesForParams = np.empty((len(params), self._eInputs.size, 2))
        for i in xrange(len(params)):
            spikeRatesForOnseSetParams = \
             self._calculateSpikeRatesForOneSetParams(oneSetParams=params[i])
            spikeRatesForParams[i, :, :] = spikeRatesForOnseSetParams
        return(spikeRatesForParams)

    def _calculateSpikeRatesForOneSetParams(self, oneSetParams):
        print("Calculating spike rates for ")
        print(oneSetParams)
        sys.stdout.flush()

        _, _, eRs, iRs = \
         self._twoPIFEDIntegrator.integrate(eRho0=self._eRho0, 
                                             iRho0=self._iRho0, 
                                             wEI=oneSetParams[0], 
                                             wIE=oneSetParams[1], 
                                             eInputs=self._eInputs)
        return(np.column_stack((eRs, iRs)))

