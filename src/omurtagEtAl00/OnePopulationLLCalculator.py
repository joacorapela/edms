
import numpy as np
import math
import pdb

class OnePopulationLLCalculator:

    def calculateLLsForParams(self, ys, spikeRatesForParams, sigma):
        lls = []
        for paramIndex in xrange(spikeRatesForParams.shape[0]):
            spikeRatesForParam = \
             np.reshape(spikeRatesForParams[paramIndex, :],
                         spikeRatesForParams.shape[1:])
            lls.append(self.calculateLL(ys=ys, spikeRates=spikeRatesForParam, 
                                               sigma=sigma))
        return(lls)

    def calculateLL(self, ys, spikeRates, sigma):
        return(-ys.size/2*math.log(2*math.pi*sigma**2)-\
                np.power(ys-spikeRates, 2).sum()/(2*sigma**2))
