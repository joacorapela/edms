
import numpy as np
import math
import pdb
from IFEnsembleDensityIntegratorRNF import IFEnsembleDensityIntegratorRNF
from ifEDMsFunctions import computeA2

class IFEnsembleDensityIntegratorRNFWI(IFEnsembleDensityIntegratorRNF):

    def __init__(self, nVSteps, leakage, hMu, hSigma, kappaMu, kappaSigma):
        super(IFEnsembleDensityIntegratorRNFWI, self).\
         __init__(nVSteps=nVSteps, leakage=leakage, hMu=hMu, hSigma=hSigma)
        self._a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu, 
                                              kappaSigma=kappaSigma)
        self._kappaMu = kappaMu
        self._kappaSigma = kappaSigma

    def prepareToIntegrate(self, t0, tf, dt, eInputCurrent, iInputCurrent):
        super(IFEnsembleDensityIntegratorRNFWI, self).\
         prepareToIntegrate(t0=t0, tf=tf, dt=dt, eInputCurrent=eInputCurrent)
        self._iInputCurrent = iInputCurrent
        nTSteps = round((self._tf-self._t0)/self._dt)
        self._iInputCurrentHist = np.zeros(nTSteps+1)

    def setInitialValue(self, rho0):
        super(IFEnsembleDensityIntegratorRNFWI, self).setInitialValue(rho0=rho0)
        self._iInputCurrentHist[0] = 0.0

    def _deriv(self, t, rho):
        sigmaE = self._eInputCurrent(t)
        binIndex = round((t-self._t0)/self._dt)
        self._eInputCurrentHist[binIndex] = sigmaE
        sigmaI = self._iInputCurrent(t=t)
        self._iInputCurrentHist[binIndex] = sigmaI
        q = self._a0+sigmaE*self._a1-sigmaI*self._a2
        answer = q.dot(rho)
        return(answer)

    def getIInputCurrentHist(self):
        return self._iInputCurrentHist

