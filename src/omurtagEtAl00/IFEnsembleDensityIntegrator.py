
import sys
import numpy as np
import math
import warnings
from scipy.integrate import ode
import pdb
import abc
import matplotlib.pyplot as plt
from ifEDMsFunctions import computeA0Tilde

class IFEnsembleDensityIntegrator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, nVSteps, leakage, hMu):
        self._a0Tilde = computeA0Tilde(nVSteps=nVSteps)
        self._leakage = leakage
        self._hMu = hMu

    @abc.abstractmethod
    def _computeSpikeRate(self, rho, t):
        return

    @abc.abstractmethod
    def _deriv(self, t, rho):
        return

    def prepareToIntegrate(self, t0, tf, dt, eExternalInput):
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._integrator = ode(self._deriv).set_integrator('vode')
        nTSteps = round((tf-t0)/dt)
        self._eExternalInput = eExternalInput
        self._eExternalInputHist = []

    def setInitialValue(self, rho0):
        self._rho0 = rho0
        self._sumRho0 = rho0.sum()
        self._integrator.set_initial_value(rho0, self._t0)
#         self._eExternalInputHist[0] = self._getEExternalInput(t=self._t0)

    def integrate(self, dtSaveRhos=None, nStepsBtwPrintouts=1000):
        '''
        Answers time series of length round((self._tf-self._t0)/self._dt)
        '''

        if dtSaveRhos==None:
            dtSaveRhos = self._dt
        nTSteps = round((self._tf-self._t0)/self._dt)
        nTStepsSaveRhos = round((self._tf-self._t0)/dtSaveRhos)
        saveRhosTimeDSFactor = round(dtSaveRhos/self._dt)
        times = []
        rhos = []
        spikeRates = []
        eExternalInputs = []
        eFeedbackInputs = []
        iExternalInputs = []
        iFeedbackInputs = []
        times.append(self._t0)
        rhos.append(self._rho0)
        spikeRate = self._computeSpikeRate(rho=self._rho0, t=self._t0)
        spikeRates.append(spikeRate)
        eExternalInputs.append(self._eExternalInput(t=self._t0))
        iExternalInputs.append(self._iExternalInput(t=self._t0))
        eFeedbackInputs.append(self._nInputsPerNeuron*self._fracExcitatoryNeurons*spikeRate)
        iFeedbackInputs.append((self._nInputsPerNeuron*(1-self._fracExcitatoryNeurons)*spikeRate))
        successfulIntegration = True
        t = self._t0
        step = 0
        stepRho = 0
        spikeRate = spikeRates[0]
        while successfulIntegration and step<nTSteps-1:
            step = step+1
            if step%nStepsBtwPrintouts==0:
                print("Processing time %.05f out of %.02f (%f)" % (t, 
                                                                    self._tf, 
                                                                    spikeRate))
                sys.stdout.flush()
                # plt.cla()
                # plt.plot(rho)
                # plt.draw()
                # pdb.set_trace()
            successfulIntegration, t, rho, spikeRate = self.integrateOneDeltaT(t=t)
            times.append(t)
            spikeRates.append(spikeRate)
            eExternalInputs.append(self._eExternalInput(t=t))
            iExternalInputs.append(self._iExternalInput(t=t))
            eFeedbackInputs.append(self._nInputsPerNeuron*self._fracExcitatoryNeurons*spikeRate)
            iFeedbackInputs.append((self._nInputsPerNeuron*(1-self._fracExcitatoryNeurons)*spikeRate))
            if step%saveRhosTimeDSFactor==0:
                stepRho = stepRho+1
                rhos.append(rho)

        timesArray = np.array(times)
        rhosArray = np.empty((len(rhos[0]), len(rhos)))
        rhosArray[:,:] = float("nan")
        for i in xrange(len(rhos)):
            rhosArray[:,i] = rhos[i]
        spikeRatesArray = np.array(spikeRates)
        eExternalInputsArray = np.array(eExternalInputs)
        eFeedbackInputsArray = np.array(eFeedbackInputs)
        iExternalInputsArray = np.array(iExternalInputs)
        iFeedbackInputsArray = np.array(iFeedbackInputs)
        return(timesArray, rhosArray, spikeRatesArray, saveRhosTimeDSFactor,
                           eExternalInputsArray, eFeedbackInputsArray, 
                           iExternalInputsArray, iFeedbackInputsArray)

    def integrateOneDeltaT(self, t):
        self._integrator.integrate(t+self._dt)
        t = self._integrator.t
        rho = self._integrator.y
        spikeRate = self._computeSpikeRate(rho=rho, t=t)
        
        return((self._integrator.successful(), t, rho, spikeRate))

    '''
    def getSpikeRate(self, t):
        if t<self._t0:
            return(0.0)
        binIndex = round((t-self._t0)/self._dt)
        spikeRate = np.nan
        if(binIndex<self._spikeRates.size):
            spikeRate = self._spikeRates[binIndex]
        if(math.isnan(spikeRate)):
            validIndices = np.where(~np.isnan(self._spikeRates))[0]
            if(len(validIndices)==0):
                raise RunimeError("No spike rate available yet")
            latestValidIndex = validIndices[-1]
            spikeRate = self._spikeRates[latestValidIndex]
            warnings.warn("Using the latest available spike rate located %d bins before the requested bin" % (binIndex-latestValidIndex))
            pdb.set_trace()
        return(spikeRate)
    '''

    def getT0(self):
        return(self._t0)

    def getTf(self):
        return(self._tf)

    def getDt(self):
        return(self._dt)

    def getRho0(self):
        return(self._rho0)

    def getSpikeRates(self):
        return(self._spikeRates)

    def getEExternalInputHist(self):
        return(self._eExternalInputHist)

    def getLeakage(self):
        return(self._leakage)

    def setLeakage(self, leakage):
        self._leakage = leakage

    def getA0Tilde(self):
        return(self._a0Tilde)
