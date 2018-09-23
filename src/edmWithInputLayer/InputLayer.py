
import pdb
import numpy as np

class InputLayer:

    def __init__(self, stimuliSource, rectification, filter, baselineInput):
        self._stimuliSource = stimuliSource
        self._rectification = rectification
        self._filter = filter
        self._baselineInput = baselineInput

    def _getProjection(self, t):
        x = self._stimuliSource.getStimulus(t=t)
        p = np.dot(x, self._filter)
        return(p)

    def getSpikeRate(self, t):
        p = self._getProjection(t=t)
        spikeRate = self._rectification.eval(x=p)
        return(spikeRate+self._baselineInput)

    def getRectification(self):
        return(self._rectification)

    def getFilter(self):
        return(self._filter)

    def setFilter(self, filter):
        self._filter = filter
