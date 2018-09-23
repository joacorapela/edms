
import numpy as np
import pdb

class LLCalculator:

    def calculateLLsForParams(self, ys, spikeRatesForParams, sigma):
        lls = []
        for paramIndex in xrange(spikeRatesForParams.shape[0]):
            spikeRatesForParam = \
             np.reshape(spikeRatesForParams[paramIndex, :, :],
                         spikeRatesForParams.shape[1:])
            lls.append(self.calculateLL(ys=ys, spikeRates=spikeRatesForParam, 
                                               sigma=sigma))
        return(lls)

    def calculateLL(self, ys, spikeRates, sigma):
        ll = 0.0
        for popIndex in xrange(spikeRates.shape[1]):
            spikeRatesForPop = spikeRates[:, popIndex]
            ysForPop = ys[:, popIndex]
            ll = ll + self._calculateLLForPopulation(ys=ysForPop,
                                                      spikeRatesForPop=
                                                       spikeRatesForPop,
                                                      sigma=sigma)
        return(ll)

    def _calculateLLForPopulation(self, ys, spikeRatesForPop, sigma):
        return(-np.power(ys-spikeRatesForPop, 2).sum()/(2*sigma**2))
