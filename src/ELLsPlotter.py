
import sys
import pdb
import math
import numpy as np
import pdb
import matplotlib.pyplot as plt

class ELLsPlotter:
    def plotELLsTimeCourses(self, eLLs, times, figFilename, 
                                  xlabel="Time (sec)", 
                                  ylabel="Parameter Index"):
        for i in xrange(eLLs.shape[0]):
            plt.plot(times, eLLs[i,:])
        plt.xlabel(xlabel, fontsize="large")
        plt.ylabel(ylabel, fontsize="large")
        plt.savefig(figFilename)
        plt.close()

    def plotELLs(self, eLLs, times, timeToPlot, params, trueParams, 
                       paramXIndex, paramYIndex, 
                       xlabel, ylabel, figFilename, nLevels=30, 
                       minELLToPlot=-float("inf"), cbLabel="Log Likelihood",
                       measurementsMark="x", measurementsColor="white",
                       trueParamsMark="o", trueParamsColor="white"):
        # self._paramsMatch(params=np.array((40.0, 1, 0.2)), 
        #                    paramXIndex=0,
        #                    paramXValue=40,
        #                    paramYIndex=1,
        #                    paramYValue=1,
        #                    trueParams=np.array(((20.0, 10, 0.2))))
        timeIndexToPlot = np.argmin(np.abs(times-timeToPlot))
        paramXValues, paramYValues, eLLsMatrix = \
         self._buildValuesMatrix(values=eLLs, timeIndexToPlot=timeIndexToPlot,
                                              params=params, 
                                              paramXIndex=paramXIndex, 
                                              paramYIndex=paramYIndex,
                                              trueParams=trueParams)
        X, Y = np.meshgrid(paramXValues, paramYValues)
        tooSmallELLIndices = np.where(eLLsMatrix<minELLToPlot)
        eLLsMatrix[tooSmallELLIndices[0], tooSmallELLIndices[1]] = minELLToPlot
        cs = plt.contourf(X, Y, eLLsMatrix, nLevels)
        plt.plot(X, Y, measurementsMark, color=measurementsColor)
        plt.plot(trueParams[paramXIndex], trueParams[paramYIndex],
                                          trueParamsMark, color=trueParamsColor)
        cb = plt.colorbar(cs)
        cb.set_label(cbLabel, fontsize="large")
        plt.xlabel(xlabel, fontsize="large")
        plt.ylabel(ylabel, fontsize="large")
        plt.savefig(figFilename)
        plt.close()

    def plotELLsAndGradients(self, eLLsAndGradients, times, timeToPlot, params, trueParams, 
                                  paramXIndex, paramYIndex, 
                                  paramXScale, paramYScale,
                                  xlabel, ylabel, figFilename, nLevels=30, 
                                  minELLToPlot=-float("inf"), 
                                  cbLabel="Log Likelihood",
                                  measurementsMark="x", 
                                  measurementsColor="white",
                                  trueParamsMark="o", 
                                  trueParamsColor="white",
                                  arrowsColor="white",
                                  xlim=None, ylim=None):
        eLLs = eLLsAndGradients[-1]
        timeIndexToPlot = np.argmin(np.abs(times-timeToPlot))
        paramXValues, paramYValues, eLLsMatrix = \
         self._buildValuesMatrix(values=eLLs, 
                                  timeIndexToPlot=timeIndexToPlot, 
                                  params=params, 
                                  paramXIndex=paramXIndex, 
                                  paramYIndex=paramYIndex,
                                  trueParams=trueParams)
        if xlim is not None:
            paramXInRange = np.logical_and(xlim[0]<=paramXValues, 
                                            paramXValues<=xlim[1])
            paramXValues = paramXValues[paramXInRange]
            eLLsMatrix = eLLsMatrix[:,paramXInRange]
        if ylim is not None:
            paramYInRange = np.logical_and(ylim[0]<=paramYValues,
                                             paramYValues<=ylim[1])
            paramYValues = paramYValues[paramYInRange]
            eLLsMatrix = eLLsMatrix[paramYInRange,:]
        X, Y = np.meshgrid(paramXValues, paramYValues)
        # tooSmallELLIndices = np.where(eLLsMatrix[:,0]<minELLToPlot)
        # eLLsMatrix[tooSmallELLIndices[0], tooSmallELLIndices[1]] = minELLToPlot

        derivativeXs = eLLsAndGradients[paramXIndex]
        _, _, derivativeXsMatrix = \
         self._buildValuesMatrix(values=derivativeXs,
                                  timeIndexToPlot=timeIndexToPlot,
                                  params=params, 
                                  paramXIndex=paramXIndex, 
                                  paramYIndex=paramYIndex,
                                  trueParams=trueParams)
        if xlim is not None:
            derivativeXsMatrix = derivativeXsMatrix[:,paramXInRange]
        if ylim is not None:
            derivativeXsMatrix = derivativeXsMatrix[paramYInRange,:]

        derivativeYs = eLLsAndGradients[paramYIndex]
        _, _, derivativeYsMatrix = \
         self._buildValuesMatrix(values=derivativeYs, 
                                  timeIndexToPlot=timeIndexToPlot, 
                                  params=params, 
                                  paramXIndex=paramXIndex, 
                                  paramYIndex=paramYIndex,
                                  trueParams=trueParams)
        if xlim is not None:
            derivativeYsMatrix = derivativeYsMatrix[:,paramXInRange]
        if ylim is not None:
            derivativeYsMatrix = derivativeYsMatrix[paramYInRange,:]
        derivativeXsMatrix = derivativeXsMatrix*paramXScale
        derivativeYsMatrix = derivativeYsMatrix*paramYScale
        N = np.sqrt(derivativeXsMatrix**2+derivativeYsMatrix**2)
        maxN = N.max()
        derivativeXsMatrix = derivativeXsMatrix/maxN
        derivativeYsMatrix = derivativeYsMatrix/maxN

        cs = plt.contourf(X, Y, eLLsMatrix, nLevels)
        plt.quiver(X, Y, derivativeXsMatrix, derivativeYsMatrix, 
                      color=arrowsColor)
        plt.plot(X, Y, measurementsMark, color=measurementsColor)
        plt.plot(trueParams[paramXIndex], trueParams[paramYIndex],
                                          trueParamsMark, color=trueParamsColor)
        cb = plt.colorbar(cs)
        cb.set_label(cbLabel, fontsize="large")
        plt.xlabel(xlabel, fontsize="large")
        plt.ylabel(ylabel, fontsize="large")
        plt.savefig(figFilename)
        plt.close()

    def _getParamValues(self, params, paramIndex):
        paramValues = []
        for i in xrange(len(params)):
            paramValues.append(params[i][paramIndex])
        return(np.sort(np.unique(np.array(paramValues))))

    def _paramsMatch(self, params, 
                           paramXIndex, paramXValue, 
                           paramYIndex, paramYValue,
                           trueParams, floatComparisonTol=1e-6):
        '''
        Answers True if params are the trueParams with the exception of
        params[paramXIndex] and params[paramYIndex] that should equal to
        paramXValue and paramYValue, respectively
        '''
        for paramIndex in xrange(params.size):
            paramValue = params[paramIndex]
            if (paramIndex==paramXIndex and \
                abs(paramValue-paramXValue)>floatComparisonTol) or\
               (paramIndex==paramYIndex and \
                abs(paramValue-paramYValue)>floatComparisonTol) or\
               (paramIndex!=paramXIndex and paramIndex!=paramYIndex and\
                abs(paramValue-trueParams[paramIndex])>floatComparisonTol):
                return(False)
        return(True)

    def _findValueForParams(self, values, timeIndexToPlot, params, 
                                  paramXIndex, paramXValue, 
                                  paramYIndex, paramYValue, trueParams):
        for i in xrange(len(params)):
            if self._paramsMatch(params=params[i], paramXIndex=paramXIndex, paramXValue=paramXValue, paramYIndex=paramYIndex, paramYValue=paramYValue, trueParams=trueParams):
                return(values[i, timeIndexToPlot])
        raise RuntimeError("Could not find value for parameters (%d=%f, %d=%f)"%\
                               (paramXIndex, paramXValue, paramYIndex, paramYValue))

    def _buildValuesMatrix(self, values, timeIndexToPlot, params, 
                                 paramXIndex, paramYIndex, trueParams):
        paramXValues = self._getParamValues(params=params, 
                                             paramIndex=paramXIndex)
        paramYValues = self._getParamValues(params=params, 
                                             paramIndex=paramYIndex)
        valuesMatrix = np.empty((paramYValues.size, paramXValues.size))

        for i in xrange(len(paramXValues)):
            for j in xrange(len(paramYValues)):
                paramXValue = paramXValues[i]
                paramYValue = paramYValues[j]
                valueForParams = self._findValueForParams(values=values, 
                                                           timeIndexToPlot=
                                                            timeIndexToPlot,
                                                           params=params,
                                                           paramXIndex=
                                                            paramXIndex,
                                                           paramXValue=
                                                            paramXValue,
                                                           paramYIndex=
                                                            paramYIndex,
                                                           paramYValue=
                                                            paramYValue,
                                                           trueParams=
                                                            trueParams)
                valuesMatrix[j, i] = valueForParams
        return(paramXValues, paramYValues, valuesMatrix)

