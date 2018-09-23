
import numpy as np
from LDIFEnsembleDensityIntegratorRWF import LDIFEnsembleDensityIntegratorRWF
from myUtils import splitRealAndImaginaryPartsInMatrix, \
 splitRealAndImaginaryPartsInVector

class LDIFEnsembleDensityIntegratorRWFNI(LDIFEnsembleDensityIntegratorRWF):

    def __init__(self, nVSteps, leakage, hMu, hSigma, nInputsPerNeuron, 
                       nEigen, eigenRepos):
        super(LDIFEnsembleDensityIntegratorRWFNI, self).\
         __init__(nVSteps=nVSteps, leakage=leakage, hMu=hMu, hSigma=hSigma,
                                   nInputsPerNeuron=nInputsPerNeuron,
                                   nEigen=nEigen,
                                   eigenRepos=eigenRepos)

    def setInitialValue(self, rho0):
        super(LDIFEnsembleDensityIntegratorRWFNI, self).\
         setInitialValue(rho0=rho0)
        self._sriLDCoefs0 = \
         self._getSRILDCoefs0(rho0=rho0, nEigen=self._nEigen)
        self._integrator.set_initial_value(self._sriLDCoefs0, self._t0)


    def _deriv(self, t, sriLDCoefs):
        eVecs, sigmaE, eInputCurrentAtT, eFeedbackCurrentAtT = \
         self._getEVecsAndCurrents(t=t)

        # Save currents
        binIndex = round((t-self._t0)/self._dt)
        self._eInputCurrentHist[binIndex] = eInputCurrentAtT
        self._eFeedbackCurrentHist[binIndex] = eFeedbackCurrentAtT

        # Get prevSigmaE and prevSigmaI
        if binIndex>0:
            prevSigmaE = self._eInputCurrentHist[binIndex-1]+\
                          self._eFeedbackCurrentHist[binIndex-1]
        else:
            prevSigmaE = sigmaE

        # Finally, with sigmaE, prevSigmaE, computer deriv
        prevEVals = self._eigenRepos.getEigenvalues(prevSigmaE)
        prevEVecs = self._eigenRepos.getEigenvectors(prevSigmaE)
        rPrevEVals = prevEVals[:self._nEigen]
        rPrevEVecs = prevEVecs[:, :self._nEigen]
        rEVecs = eVecs[:, :self._nEigen]

        diagRPrevEVals = np.diag(1+self._dt*rPrevEVals)
        diffMatrix = (np.linalg.pinv(rEVecs).dot(rPrevEVecs).\
                      dot(diagRPrevEVals)-np.identity(self._nEigen))/self._dt
        sriDiffMatrix = splitRealAndImaginaryPartsInMatrix(m=diffMatrix)
        deriv =  sriDiffMatrix.dot(sriLDCoefs)
        return(deriv)

    def _getExcitatoryInputCurrent(self, t):
        sigmaE = self._eInputCurrent(t=t)+(self._nInputsPerNeuron*
                                            self.getSpikeRate(t=t-self._dt))
        return(sigmaE)

    def _getSRILDCoefs0(self, rho0, nEigen):
        sigmaE = self._eInputCurrent(t=self._t0)
        eVecs = self._eigenRepos.getEigenvectors(sigmaE)
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
        spikeRateAtTMinusDT = self.getSpikeRate(t=t-self._dt)

        eFeedbackCurrentAtT = (self._nInputsPerNeuron*spikeRateAtTMinusDT)
        eInputCurrentAtT = self._eInputCurrent(t=t)
        sigmaE = eInputCurrentAtT + eFeedbackCurrentAtT

        eVecs = self._eigenRepos.getEigenvectors(sigmaE)

        return((eVecs, sigmaE, eInputCurrentAtT, eFeedbackCurrentAtT))

    def _getEVecsAndSigmaE(self, t):
        eVecs, sigmaE, _, _ = self._getEVecsAndCurrents(t=t)
        return(eVecs, sigmaE)

