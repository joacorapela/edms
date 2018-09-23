
import numpy as np
from scipy.special import erfc
import math
import matplotlib.pylab as plt
import pdb
from LDIFEnsembleDensityIntegratorR import LDIFEnsembleDensityIntegratorR
from myUtils import splitRealAndImaginaryPartsInMatrix, \
 splitRealAndImaginaryPartsInVector
from ifEDMsFunctions import computeA2

class LDIFEnsembleDensityIntegratorRNFWI(LDIFEnsembleDensityIntegratorR):

    def __init__(self, nVSteps, leakage, hMu, hSigma, kappaMu, kappaSigma,
                       nEigen, eigenRepos):
        super(LDIFEnsembleDensityIntegratorRNFWI, self).\
         __init__(nVSteps=nVSteps, leakage= leakage, hMu=hMu, hSigma=hSigma,
                                   nEigen=nEigen, eigenRepos=eigenRepos)
        self._a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu,
                                              kappaSigma=kappaSigma)
        self._kappaMu = kappaMu
        self._kappaSigma = kappaSigma


    def prepareToIntegrate(self, t0, tf, dt, spikeRate0, 
                                 eInputCurrent, iInputCurrent):
        super(LDIFEnsembleDensityIntegratorRNFWI, self).\
         prepareToIntegrate(t0=t0, tf=tf, dt=dt, spikeRate0=spikeRate0,
                                   eInputCurrent=eInputCurrent)
        self._iInputCurrent = iInputCurrent
        nTSteps = round((self._tf-self._t0)/self._dt)
        self._iInputCurrentHist = np.zeros(nTSteps+1)
        self._iInputCurrentHist[0] = 0.0

    def setInitialValue(self, rho0):
        super(LDIFEnsembleDensityIntegratorRNFWI, self).\
         setInitialValue(rho0=rho0)
        self._sriLDCoefs0 = \
         self._getSRILDCoefs0(rho0=rho0, nEigen=self._nEigen)
        self._integrator.set_initial_value(self._sriLDCoefs0, self._t0)

    def _deriv(self, t, sriLDCoefs):
        eVecs, sigmaE, sigmaI = self._getEVecsAndCurrents(t=t)

        # Save currents
        binIndex = round((t-self._t0)/self._dt)
        self._eInputCurrentHist[binIndex] = sigmaE
        self._iInputCurrentHist[binIndex] = sigmaI

        # Get prevSigmaE and prevSigmaI
        if binIndex>0:
            prevSigmaE = self._eInputCurrentHist[binIndex-1]
            prevSigmaI = self._iInputCurrentHist[binIndex-1]
        else:
            prevSigmaE = sigmaE
            prevSigmaI = sigmaI

        # Finally, with sigmaE, prevSigmaE, sigmaI, prevSigmaI computer deriv
        prevEVals = self._eigenRepos.getEigenvalues(sE=prevSigmaE,
                                                     sI=prevSigmaI)
        prevEVecs = self._eigenRepos.getEigenvectors(sE=prevSigmaE,
                                                      sI=prevSigmaI)
        rPrevEVals = prevEVals[:self._nEigen]
        rPrevEVecs = prevEVecs[:, :self._nEigen]
        rEVecs = eVecs[:, :self._nEigen]

        diagRPrevEVals = np.diag(1+self._dt*rPrevEVals)
        diffMatrix = (np.linalg.pinv(rEVecs).dot(rPrevEVecs).\
                      dot(diagRPrevEVals)-np.identity(self._nEigen))/self._dt
        sriDiffMatrix = splitRealAndImaginaryPartsInMatrix(m=diffMatrix)
        deriv =  sriDiffMatrix.dot(sriLDCoefs)

        # begin remove
#         print("t=%f: sigmaE=%f, prevSigmaE=%f, sigmaI=%f, prevSigmaI=%f" % (t, sigmaE, prevSigmaE, sigmaI, prevSigmaI))
#         plt.figure()
#         plt.plot(rPrevEVecs[:, 0])
#         plt.figure()
#         plt.plot(rPrevEVals)
#         plt.show()
#         pdb.set_trace()
#         plt.close("all")
        # end remove

        return(deriv)

    def _getExcitatoryInputCurrent(self, t):
        sigmaE = self._eInputCurrent(t=t)
        return(sigmaE)

    def getIInputCurrentHist(self):
        return self._iInputCurrentHist

    def _getSRILDCoefs0(self, rho0, nEigen):
        sigmaE = self._eInputCurrent(t=self._t0)
        sigmaI = self._iInputCurrent(t=self._t0)
        eVecs = self._eigenRepos.getEigenvectors(sE=sigmaE, sI=sigmaI)
        aEVecs = np.linalg.pinv(eVecs).transpose().conjugate()
        rEVecs = eVecs[:, :nEigen]
        rAEVecs = aEVecs[:, :nEigen]
        ldCoefs = rAEVecs.transpose().conjugate().dot(rho0)
        sriLDCoefs = splitRealAndImaginaryPartsInVector(v=ldCoefs)
        # To avoid loosing probability, lets enforce 
        # coefsFromDE[0]=sum(rho0)/surm(\phi_0) (see Note 12 in knigth00)
        sriLDCoefs[0] = sum(rho0)/sum(rEVecs[:, 0].real)
        sriLDCoefs[1] = 0

        return(sriLDCoefs)

    def _getEVecsAndCurrents(self, t):
        # Compute sigmaE
        if t-self._dt>self._t0:
            spikeRateAtTMinusDT = self.getSpikeRate(t=t-self._dt)
        else:
            spikeRateAtTMinusDT = 0.0

        eInputCurrentAtT = self._eInputCurrent(t=t)
        sigmaE = eInputCurrentAtT

        # Compute sigmaI and save inhibitory currents
        sigmaI = iInputCurrentAtT = self._iInputCurrent(t=t)
        eVecs = self._eigenRepos.getEigenvectors(sE=sigmaE, sI=sigmaI)
        return((eVecs, sigmaE, sigmaI))

    def _getEVecsAndSigmaE(self, t):
        eVecs, sigmaE, _ = self._getEVecsAndCurrents(t=t)
        return(eVecs, sigmaE)

