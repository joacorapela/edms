
import sys
import numpy as np
import math
import warnings
from scipy.integrate import ode
import pdb
import abc
from ifEDMsFunctions import computeA0
from EulerIntegrator import EulerIntegrator

class LDIFEnsembleDensityIntegrator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, nVSteps, leakage, nEigen, eigenRepos):
        self._a0 = computeA0(nVSteps=nVSteps, leakage=leakage)
        self._nEigen = nEigen
        self._eigenRepos = eigenRepos

    @abc.abstractmethod
    def _getEVecsAndSigmaE(self, t):
        return

    @abc.abstractmethod
    def _computeSpikeRate(self, ldCoefs, t):
        return

    @abc.abstractmethod
    def _getExcitatoryInputCurrent(self, t):
        return

    def prepareToIntegrate(self, t0, tf, dt, eInputCurrent):
        self._t0 = t0
        self._tf = tf
        self._dt = dt
#         self._integrator = ode(self._deriv).set_integrator('dopri5')
#                                                              nsteps=1,
#                                                              safety=0,
#                                                              max_step=dt
#                                                              rtol=1e-4,
#                                                              atol=1e-6
#                                                            )
        self._integrator = EulerIntegrator(deriv=self._deriv, dt=dt)
        self._eInputCurrent = eInputCurrent
        nTSteps = round((tf-t0)/dt)
        self._spikeRates = np.empty(nTSteps+1)
        self._spikeRates[:] = np.nan
        self._eInputCurrentHist = np.zeros(nTSteps+1)

    def setInitialValue(self, rho0):
        self._sumRho0 = rho0.sum()

        nVSteps = self._a0.shape[0]
        dv = 1.0/nVSteps
        sigma0E = self._eInputCurrentHist[0]
        spikeRate0 = sigma0E*dv*self._reversedQs.dot(rho0)
        self._spikeRates[0] = spikeRate0
        self._prevSpikeRate = spikeRate0
        self._eInputCurrentHist[0] = self._eInputCurrent(t=self._t0)

    def integrate(self, dtSaveLDCoefs):
        nTSteps = round((self._tf-self._t0)/self._dt)
        nTStepsSaveLDCoefs = round((self._tf-self._t0)/dtSaveLDCoefs)
        saveLDCoefsTimeDSFactor = round(dtSaveLDCoefs/self._dt)
        times = np.empty(nTSteps+1)
        times[:] = np.nan
        sriLDCoefsCol = np.empty((2*self._nEigen, nTStepsSaveLDCoefs+1))
        sriLDCoefsCol[:] = np.nan
        times[0] = self._t0
        sriLDCoefsCol[:, 0] = self._sriLDCoefs0
        successfulIntegration = True
        t = self._t0
        sriLDCoefs = self._sriLDCoefs0
        step = 0
        stepLDCoefs = 0
        while successfulIntegration and step<nTSteps:
            step = step+1
            if step%1000==0:
                print("Processing time %.05f out of %.02f (spike rate=%f)" %
                      (t, self._tf, spikeRate))
                sys.stdout.flush()
            successfulIntegration, t, sriLDCoefs, spikeRate = \
             self.integrateOneDeltaT(t=t, sriLDCoefs=sriLDCoefs)
            times[step] = t
            if step%saveLDCoefsTimeDSFactor==0:
                stepLDCoefs = stepLDCoefs+1
                sriLDCoefsCol[:, stepLDCoefs] = sriLDCoefs
        return(times, sriLDCoefsCol, self._spikeRates, saveLDCoefsTimeDSFactor)

    def integrateOneDeltaT(self, t, sriLDCoefs):
#         self._integrator.integrate(t+self._dt)
#         t = self._integrator.t
#         sriLDCoefs = self._integrator.y

        t, sriLDCoefs, rEVecs, sigmaE = self._getNextValues(t, sriLDCoefs)

        # To avoid loosing probability, lets enforce 
        # coefsFromDE[0]=sum(rho0)/surm(\phi_0) (see Note 12 in knigth00)
        sriLDCoefs[0] = self._sumRho0/sum(rEVecs[:, 0].real)
        sriLDCoefs[1] = 0.0
        self._integrator.y = sriLDCoefs

        spikeRate = self._computeSpikeRate(sriLDCoefs=sriLDCoefs, 
                                            rEVecs=rEVecs, 
                                            sigmaE=sigmaE)
        self._prevSpikeRate = spikeRate

        index = round((t-self._t0)/self._dt)
        self._spikeRates[index] = spikeRate

        return((True, t, sriLDCoefs, spikeRate))

    def getSpikeRate(self, t):
        if t<self._t0:
            return 0.0
        binIndex = round((t-self._t0)/self._dt)
        spikeRate = np.nan
        if(binIndex<self._spikeRates.size):
            spikeRate = self._spikeRates[binIndex]
        if(math.isnan(spikeRate)):
            validIndices = np.where(~np.isnan(self._spikeRates))[0]
            if(len(validIndices)==0):
                pdb.set_trace()
                raise RuntimeError("No spike rate available yet")
            latestValidIndex = validIndices[-1]
            spikeRate = self._spikeRates[latestValidIndex]
            warnings.warn("Using the latest available spike rate at bin %d instead of the requested one at bin %d" % (latestValidIndex, binIndex))
            pdb.set_trace()
        return(spikeRate)

    def getT0(self):
        return(self._t0)

    def getTf(self):
        return(self._tf)

    def getDt(self):
        return(self._dt)

    def getSRILDCoefs(self):
        return(self._sriLDCoefs0)

    def getNEigen(self):
        return(self._nEigen)

    def getSpikeRates(self):
        return(self._spikeRates)

    def getEInputCurrentHist(self):
        return(self._eInputCurrentHist)

    def getEFeedbackCurrentHist(self):
        return(self._eFeedbackCurrentHist)

