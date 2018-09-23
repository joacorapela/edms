
import numpy as np
from scipy.special import erfc
import math
import matplotlib.pylab as plt
import pdb
from LDIFEnsembleDensityIntegratorRWF import LDIFEnsembleDensityIntegratorRWF
from myUtils import splitRealAndImaginaryPartsInMatrix, \
 splitRealAndImaginaryPartsInVector
from ifEDMsFunctions import computeA2

class LDIFEnsembleDensityIntegratorRWFWI(LDIFEnsembleDensityIntegratorRWF):

    def __init__(self, nVSteps, leakage, hMu, hSigma, kappaMu, kappaSigma,
                       fracExcitatoryNeurons, nInputsPerNeuron,
                       nEigen, eigenRepos):
        super(LDIFEnsembleDensityIntegratorRWFWI, self).\
         __init__(nVSteps=nVSteps, leakage= leakage, hMu=hMu, hSigma=hSigma,
                                   nInputsPerNeuron=nInputsPerNeuron,
                                   nEigen=nEigen, eigenRepos=eigenRepos)
        self._a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu,
                                              kappaSigma=kappaSigma)
        self._kappaMu = kappaMu
        self._kappaSigma = kappaSigma
        self._fracExcitatoryNeurons = fracExcitatoryNeurons


    def prepareToIntegrate(self, t0, tf, dt, 
                                 eInputCurrent, iInputCurrent):
        super(LDIFEnsembleDensityIntegratorRWFWI, self).\
         prepareToIntegrate(t0=t0, tf=tf, dt=dt,
                                   eInputCurrent=eInputCurrent)
        self._iInputCurrent = iInputCurrent
        nTSteps = round((self._tf-self._t0)/self._dt)
        self._iInputCurrentHist = np.zeros(nTSteps+1)
        self._iFeedbackCurrentHist = np.zeros(nTSteps+1)
        self._iInputCurrentHist[0] = 0.0
        self._iFeedbackCurrentHist[0] = 0.0              

    def setInitialValue(self, rho0):
        super(LDIFEnsembleDensityIntegratorRWFWI, self).\
         setInitialValue(rho0=rho0)
        self._sriLDCoefs0 = \
         self._getSRILDCoefs0(rho0=rho0, nEigen=self._nEigen)
        self._integrator.set_initial_value(self._sriLDCoefs0, self._t0)

    def _deriv(self, t, sriLDCoefs):
        eVecs, eVals, sigmaE, sigmaI, eInputCurrentAtT, eFeedbackCurrentAtT, \
         iInputCurrentAtT, iFeedbackCurrentAtT = self._getEigenAndCurrents(t=t)
        rEVecs = eVecs[:, :self._nEigen]
        rEVals = eVals[:self._nEigen]

        # Save currents
        binIndex = round((t-self._t0)/self._dt)
        self._eInputCurrentHist[binIndex] = eInputCurrentAtT
        self._eFeedbackCurrentHist[binIndex] = eFeedbackCurrentAtT
        self._iFeedbackCurrentHist[binIndex] = iFeedbackCurrentAtT
        if iInputCurrentAtT is not None:
            self._iInputCurrentHist[binIndex] = iInputCurrentAtT

        # Get prevSigmaE and prevSigmaI
#         if binIndex>0:
#             prevSigmaE = self._eInputCurrentHist[binIndex-1]+\
#                           self._eFeedbackCurrentHist[binIndex-1]
#             prevSigmaI = self._iFeedbackCurrentHist[binIndex-1]
#             if self._iInputCurrent is not None:
#                 prevSigmaI = prevSigmaI + self._iInputCurrentHist[binIndex-1]
#         else:
#             prevSigmaE = sigmaE
#             prevSigmaI = sigmaI

        # Finally, with sigmaE, prevSigmaE, sigmaI, prevSigmaI computer deriv
#         prevEVals = self._eigenRepos.getEigenvalues(sE=prevSigmaE,
#                                                      sI=prevSigmaI)
#         prevEVecs = self._eigenRepos.getEigenvectors(sE=prevSigmaE,
#                                                       sI=prevSigmaI)
#         prevREVals = prevEVals[:self._nEigen]
#         prevREVecs = prevEVecs[:, :self._nEigen]
        prevREVals = self._prevREVals
        prevREVecs = self._prevREVecs

        diagPrevREVals = np.diag(1+self._dt*prevREVals)
        diffMatrix = (np.linalg.pinv(rEVecs).dot(prevREVecs).\
                      dot(diagPrevREVals)-np.identity(self._nEigen))/self._dt
        sriDiffMatrix = splitRealAndImaginaryPartsInMatrix(m=diffMatrix)
        deriv =  sriDiffMatrix.dot(sriLDCoefs)

        self.prevREVecs = rEVecs
        self.prevREVals = rEVals

        # begin remove
#         print("t=%f: sigmaE=%f, prevSigmaE=%f, sigmaI=%f, prevSigmaI=%f" % (t, sigmaE, prevSigmaE, sigmaI, prevSigmaI))
#         plt.figure()
#         plt.plot(prevREVecs[:, 0])
#         plt.figure()
#         plt.plot(prevREVals)
#         plt.show()
#         pdb.set_trace()
#         plt.close("all")
        # end remove

        return(deriv)

    def _getNextValues(self, t, sriLDCoefs):
        t = t + self._dt
        eVecs, eVals, sigmaE, sigmaI, eInputCurrentAtT, eFeedbackCurrentAtT, \
         iInputCurrentAtT, iFeedbackCurrentAtT = self._getEigenAndCurrents(t=t)
        rEVecs = eVecs[:, :self._nEigen]
        rEVals = eVals[:self._nEigen]

        # Save currents
        binIndex = round((t-self._t0)/self._dt)
        self._eInputCurrentHist[binIndex] = eInputCurrentAtT
        self._eFeedbackCurrentHist[binIndex] = eFeedbackCurrentAtT
        self._iFeedbackCurrentHist[binIndex] = iFeedbackCurrentAtT
        if iInputCurrentAtT is not None:
            self._iInputCurrentHist[binIndex] = iInputCurrentAtT
        #

        diagPrevREVals = np.diag(1+self._dt*self._prevREVals)
        diffMatrix = np.linalg.pinv(rEVecs).dot(self._prevREVecs).\
                      dot(diagPrevREVals)
        sriDiffMatrix = splitRealAndImaginaryPartsInMatrix(m=diffMatrix)
        sriLDCoefs =  sriDiffMatrix.dot(sriLDCoefs)

        # Save rEVecs and rEVals for next iteration
        self._prevREVals = rEVals
        self._prevREVecs = rEVecs

        return(t, sriLDCoefs, rEVecs, sigmaE)

    def _getExcitatoryInputCurrent(self, t):
        sigmaE = self._eInputCurrent(t=t)+(self._nInputsPerNeuron*
                                            self._fracExcitatoryNeurons*
                                            self.getSpikeRate(t=t-self._dt))
        return(sigmaE)

    def getIInputCurrentHist(self):
        return self._iInputCurrentHist

    def getIFeedbackCurrentHist(self):
        return self._iFeedbackCurrentHist

    def _getSRILDCoefs0(self, rho0, nEigen):
        sigmaE = self._eInputCurrent(t=self._t0)
        if self._iInputCurrent is not None:
            sigmaI = self._iInputCurrent(t=self._t0)
        else:
            sigmaI = 0.0
        eVecs = self._eigenRepos.getEigenvectors(sE=sigmaE, sI=sigmaI)
        eVals = self._eigenRepos.getEigenvalues(sE=sigmaE, sI=sigmaI)
        aEVecs = np.linalg.pinv(eVecs).transpose().conjugate()
        rEVecs = eVecs[:, :nEigen]
        rEVals = eVals[:nEigen]
        rAEVecs = aEVecs[:, :nEigen]
        ldCoefs = rAEVecs.transpose().conjugate().dot(rho0)
        sriLDCoefs = splitRealAndImaginaryPartsInVector(v=ldCoefs)
        # To avoid loosing probability, lets enforce 
        # coefsFromDE[0]=sum(rho0)/surm(\phi_0) (see Note 12 in knigth00)
        sriLDCoefs[0] = sum(rho0)/sum(rEVecs[:, 0].real)
        sriLDCoefs[1] = 0
        self._prevREVals = rEVals
        self._prevREVecs = rEVecs

        return(sriLDCoefs)

    def _getEigenAndCurrents(self, t):
        # Compute sigmaE
        eFeedbackCurrentAtT = (self._nInputsPerNeuron*
                                self._fracExcitatoryNeurons*
                                self._prevSpikeRate)
        eInputCurrentAtT = self._eInputCurrent(t=t)
        sigmaE = eInputCurrentAtT + eFeedbackCurrentAtT

        # Compute sigmaI and save inhibitory currents
        iFeedbackCurrentAtT = (self._nInputsPerNeuron*
                                (1-self._fracExcitatoryNeurons)*
                                self._prevSpikeRate)
        sigmaI = iFeedbackCurrentAtT
        if self._iInputCurrent is not None:
            iInputCurrentAtT = self._iInputCurrent(t=t)
            sigmaI = sigmaI + iInputCurrentAtT
        else:
            iInputCurrentAtT = None
        eVecs = self._eigenRepos.getEigenvectors(sE=sigmaE, sI=sigmaI)
        eVals = self._eigenRepos.getEigenvalues(sE=sigmaE, sI=sigmaI)
        return((eVecs, eVals, sigmaE, sigmaI, eInputCurrentAtT, eFeedbackCurrentAtT, iInputCurrentAtT, iFeedbackCurrentAtT))

    def _getEVecsAndSigmaE(self, t):
        eVecs, _, sigmaE, _, _, _, _, _ = self._getEigenAndCurrents(t=t)
        return(eVecs, sigmaE)

