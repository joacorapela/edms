
import pdb
import numpy as np
from StimuliSourceFromFile import StimuliSourceFromFile
from utilsMath import Logistic
# from utilsMath import LinearFunction
from InputLayer import InputLayer

class SpikeRatesForParamsCalculatorOfInputLayer:
    def __init__(self, t0, tf, dt, stimuliFilename):
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._stimuliFilename = stimuliFilename
        self._times = np.arange(t0, tf+dt, dt)

    def calculateSpikeRatesForParams(self, params):
        nTSteps = round((self._tf-self._t0)/self._dt)
        spikeRatesForParams = np.empty((len(params), nTSteps+1))
        for i in xrange(len(params)):
            times, spikeRatesForOnseSetParams = \
             self._calculateSpikeRatesForOneSetParams(oneSetParams=params[i])
            spikeRatesForParams[i, :] = spikeRatesForOnseSetParams
        return(times, spikeRatesForParams)

    def _calculateSpikeRatesForOneSetParams(self, oneSetParams):
        l = oneSetParams[0]
        k = oneSetParams[1]
        x0 = oneSetParams[2]
        filterLength = len(oneSetParams)-3
        filter = np.array(oneSetParams[3:(3+filterLength)])
        print("Calculating spike rates for l=%.02f, k=%.02f, x0=%.02f"%\
              (l, k, x0))
        print("Filter: %s"%(str(filter)))

        stimSource = StimuliSourceFromFile(stimuliFilename=
                                             self._stimuliFilename, 
                                            t0=self._t0, tf=self._tf, 
                                            dt=self._dt)
        rectification = Logistic(k=k, x0=x0, l=l)

        inputLayer = InputLayer(stimuliSource=stimSource, 
                                 rectification=rectification,
                                 filter=filter,
                                 baselineInput=0.0)
        spikeRates = np.empty(len(self._times))
        spikeRates[:] = np.nan
        for i in xrange(len(self._times)):
            spikeRates[i] = inputLayer.getSpikeRate(t=self._times[i])
        return(self._times, spikeRates)

