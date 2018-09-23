
import numpy as np
import pdb
from edmsMath import calculateLL

class TwoPopulationsLLCalculator:

    def calculateLLsForParams(self, ys, spikeRatesForParams, ysSigma):
        lls = []
        for paramIndex in xrange(spikeRatesForParams.shape[0]):
            spikeRatesForParam = \
             np.reshape(spikeRatesForParams[paramIndex, :, :],
                         spikeRatesForParams.shape[1:])
            lls.append(self.calculateLLForPopulations(ys=ys, 
                                                       spikeRates=
                                                        spikeRatesForParam, 
                                                       ysSigma=ysSigma))
        return(lls)

    def calculateLLForPopulations(self, ys, spikeRates, ysSigma):
        ll = 0.0
        for popIndex in xrange(spikeRates.shape[1]):
            spikeRatesForPop = spikeRates[:, popIndex]
            ysForPop = ys[:, popIndex]
            ll = ll + calculateLL(ys=ysForPop, rs=spikeRatesForPop, ysSigma=ysSigma)
        return(ll)

