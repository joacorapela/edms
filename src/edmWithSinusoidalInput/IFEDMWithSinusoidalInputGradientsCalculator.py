
import sys
import pdb
import math
import numpy as np
import matplotlib.pyplot as plt
from utilsMath import Logistic, sign
from ifEDMsFunctions import buildRho0
from Sinusoidal import Sinusoidal
from BetaRho0Calculator import BetaRho0Calculator

# Begin delete
# import matplotlib.pyplot as plt
# End delete

class IFEDMWithSinusoidalInputGradientsCalculator:

    def __init__(self, a0Tilde, a1, a2, reversedQs, variablesToOptimize,
                       fixedParameterValues):
        self._a0Tilde = a0Tilde
        self._a1 = a1
        self._a2 = a2
        self._qR = reversedQs
        self._variablesToOptimize = variablesToOptimize
        self._setFixedParameterValues(values=fixedParameterValues,
                                       variablesToOptimize=variablesToOptimize)

    def deriv(self, x, ys, rs, rhos, ysSigma, times, nStepsBtwPrintouts):
        self._setParameterValuesToOptimize(values=x,
                                            variablesToOptimize=\
                                             self._variablesToOptimize)
        dt = times[1]-times[0]
        nTimeSteps = ys.size
        nVSteps = self._a0Tilde.shape[0]
        dv = 1.0/nVSteps
        idM = np.identity(nVSteps)
        sigma0E = self._sigma(dc=self._eDC,
                               ampl=self._eAmpl,
                               freq=self._eFreq,
                               phase=self._ePhase,
                               t=times[0])
        sigma0I = self._sigma(dc=self._iDC,
                               ampl=self._iAmpl,
                               freq=self._iFreq,
                               phase=self._iPhase,
                               t=times[0])
        self._initDerivVariables(dt=dt,
                                  nTimeSteps=nTimeSteps,
                                  nVSteps=nVSteps,
                                  variablesToOptimize=self._variablesToOptimize,
                                  sigma0E=sigma0E, sigma0I=sigma0I,
                                  y0=ys[0], r0=rs[0])

        for n in xrange(1, nTimeSteps):
            '''
            Derivatives of r are computed for time n
            Derivatives of rho are computed for time n+1
            '''
            # '''
            if n%nStepsBtwPrintouts==0:
                print("Gradient calculator step %d (out of %d)" % \
                      (n, nTimeSteps))
                if "leakage" in self._variablesToOptimize:
                    print "DLLLeakage=%f, "%self._dLLLeakages[n-1],
                if "g" in self._variablesToOptimize:
                    print "DLLG=%f, "%self._dLLGs[n-1],
                if "f" in self._variablesToOptimize:
                    print "DLLF=%f, "%self._dLLFs[n-1],
                if "rho0A" in self._variablesToOptimize:
                    print "DLLRho0A=%f, "%self._dLLRho0As[n-1],
                if "rho0B" in self._variablesToOptimize:
                    print "DLLRho0B=%f, "%self._dLLRho0Bs[n-1],
                if "eDC" in self._variablesToOptimize:
                    print "DLLEDC=%f, "%self._dLLEDCs[n-1],
                if "eAmpl" in self._variablesToOptimize:
                    print "DLLEAmpl=%f, "%self._dLLEAmpls[n-1],
                if "iDC" in self._variablesToOptimize:
                    print "DLLIDC=%f, "%self._dLLIDCs[n-1],
                if "iAmpl" in self._variablesToOptimize:
                    print "DLLIAmpl=%f, "%self._dLLIAmpls[n-1],
                if "iFreq" in self._variablesToOptimize:
                    print "DLLIFreq=%f, "%self._dLLIFreqs[n-1],
                if "iPhase" in self._variablesToOptimize:
                    print "DLLIPhase=%f, "%self._dLLIPhases[n-1],
                print
                sys.stdout.flush()
            # '''

            rho = rhos[:,n]

            sigma0E = self._sigma(dc=self._eDC,
                                   ampl=self._eAmpl,
                                   freq=self._eFreq,
                                   phase=self._ePhase,
                                   t=times[n])
            sigma0I = self._sigma(dc=self._iDC,
                                   ampl=self._iAmpl,
                                   freq=self._iFreq,
                                   phase=self._iPhase,
                                   t=times[n])

            qRDotRho = dv*np.dot(self._qR, rho)
            cFactorDr = 1-self._g*self._f*qRDotRho
            cFactorDRho = idM+\
                           dt*(self._leakage/2*self._a0Tilde+\
                                      (sigma0E+self._g*self._f*rs[n])*self._a1-\
                                      (sigma0I+self._g*(1-self._f)*rs[n])*\
                                      self._a2)

            if "leakage" in self._variablesToOptimize:
                qRDotDRhoLeakage = dv*np.dot(self._qR, self._dRhoLeakage)
                uLLLeakage = self._getUpdateDLLLeakage(qRDotDRhoLeakage=
                                                         qRDotDRhoLeakage,
                                                        qRDotRho=qRDotRho,   
                                                        cFactorDr=cFactorDr,
                                                        cFactorDRho=cFactorDRho,
                                                        sigma0E=sigma0E,
                                                        y=ys[n],
                                                        r=rs[n],
                                                        dt=dt,
                                                        rho=rho)
                self._dLLLeakages[n] = self._dLLLeakages[n-1] + uLLLeakage

            if "g" in self._variablesToOptimize:
                qRDotDRhoG = dv*np.dot(self._qR, self._dRhoG)
                uLLG = self._getUpdateDLLG(qRDotDRhoG=qRDotDRhoG,
                                            qRDotRho=qRDotRho, 
                                            cFactorDr=cFactorDr,
                                            cFactorDRho=cFactorDRho,
                                            sigma0E=sigma0E,
                                            y=ys[n],
                                            r=rs[n],
                                            dt=dt,
                                            rho=rho)
                self._dLLGs[n] = self._dLLGs[n-1] + uLLG

            if "f" in self._variablesToOptimize:
                qRDotDRhoF = dv*np.dot(self._qR, self._dRhoF)
                uLLF = self._getUpdateDLLF(qRDotDRhoF=qRDotDRhoF,
                                            qRDotRho=qRDotRho, 
                                            cFactorDr=cFactorDr,
                                            cFactorDRho=cFactorDRho,
                                            sigma0E=sigma0E,
                                            y=ys[n],
                                            r=rs[n],
                                            dt=dt,
                                            rho=rho)
                self._dLLFs[n] = self._dLLFs[n-1] + uLLF

            if "rho0A" in self._variablesToOptimize:
                qRDotDRhoRho0A = dv*np.dot(self._qR, self._dRhoRho0A)
                uLLRho0A = self._getUpdateDLLRho0A(qRDotDRhoRho0A=
                                                     qRDotDRhoRho0A,
                                                    qRDotRho=qRDotRho, 
                                                    cFactorDr=cFactorDr,
                                                    cFactorDRho=cFactorDRho,
                                                    sigma0E=sigma0E,
                                                    y=ys[n],
                                                    r=rs[n],
                                                    dt=dt,
                                                    rho=rho)
                self._dLLRho0As[n] = self._dLLRho0As[n-1] + uLLRho0A

            if "rho0B" in self._variablesToOptimize:
                qRDotDRhoRho0B = dv*np.dot(self._qR, self._dRhoRho0B)
                uLLRho0B = self._getUpdateDLLRho0B(qRDotDRhoRho0B=
                                                     qRDotDRhoRho0B,
                                                    qRDotRho=qRDotRho, 
                                                    cFactorDr=cFactorDr,
                                                    cFactorDRho=cFactorDRho,
                                                    sigma0E=sigma0E,
                                                    y=ys[n],
                                                    r=rs[n],
                                                    dt=dt,
                                                    rho=rho)
                self._dLLRho0Bs[n] = self._dLLRho0Bs[n-1] + uLLRho0B

            if "eDC" in self._variablesToOptimize:
                qRDotDRhoEDC = dv*np.dot(self._qR, self._dRhoEDC)
                uLLEDC = self._getUpdateDLLEDC(qRDotDRhoEDC=
                                                 qRDotDRhoEDC,
                                                qRDotRho=qRDotRho,  
                                                cFactorDr=cFactorDr,
                                                cFactorDRho=cFactorDRho,
                                                sigma0E=sigma0E,
                                                y=ys[n],
                                                r=rs[n],
                                                dt=dt,
                                                t=times[n],
                                                rho=rho)
                self._dLLEDCs[n] = self._dLLEDCs[n-1] + uLLEDC

            if "eAmpl" in self._variablesToOptimize:
                qRDotDRhoEAmpl = dv*np.dot(self._qR, self._dRhoEAmpl)
                uLLEAmpl = self._getUpdateDLLEAmpl(qRDotDRhoEAmpl=
                                                     qRDotDRhoEAmpl,
                                                    qRDotRho=qRDotRho, 
                                                    cFactorDr=cFactorDr,
                                                    cFactorDRho=cFactorDRho,
                                                    sigma0E=sigma0E,
                                                    y=ys[n],
                                                    r=rs[n],
                                                    dt=dt,
                                                    t=times[n],
                                                    rho=rho)
                self._dLLEAmpls[n] = self._dLLEAmpls[n-1] + uLLEAmpl

            if "eFreq" in self._variablesToOptimize:
                qRDotDRhoEFreq = dv*np.dot(self._qR, self._dRhoEFreq)
                uLLEFreq = self._getUpdateDLLEFreq(qRDotDRhoEFreq=
                                                     qRDotDRhoEFreq,
                                                    qRDotRho=qRDotRho, 
                                                    cFactorDr=cFactorDr,
                                                    cFactorDRho=cFactorDRho,
                                                    sigma0E=sigma0E,
                                                    y=ys[n],
                                                    r=rs[n],
                                                    dt=dt,
                                                    t=times[n],
                                                    rho=rho)
                self._dLLEFreqs[n] = self._dLLEFreqs[n-1] + uLLEFreq

            if "ePhase" in self._variablesToOptimize:
                qRDotDRhoEPhase = dv*np.dot(self._qR, self._dRhoEPhase)
                uLLEPhase = self._getUpdateDLLEPhase(qRDotDRhoEPhase=
                                                       qRDotDRhoEPhase,
                                                      qRDotRho=qRDotRho, 
                                                      cFactorDr=cFactorDr,
                                                      cFactorDRho=cFactorDRho,
                                                      sigma0E=sigma0E,
                                                      y=ys[n],
                                                      r=rs[n],
                                                      dt=dt,
                                                      t=times[n],
                                                      rho=rho)
                self._dLLEPhases[n] = self._dLLEPhases[n-1] + uLLEPhase

            if "iDC" in self._variablesToOptimize:
                qRDotDRhoIDC = dv*np.dot(self._qR, self._dRhoIDC)
                uLLIDC = self._getUpdateDLLIDC(qRDotDRhoIDC=qRDotDRhoIDC,
                                                qRDotRho=qRDotRho,
                                                cFactorDr=cFactorDr,
                                                cFactorDRho=cFactorDRho,
                                                sigma0E=sigma0E,
                                                y=ys[n],
                                                r=rs[n],
                                                dt=dt,
                                                t=times[n],
                                                rho=rho)
                self._dLLIDCs[n] = self._dLLIDCs[n-1] + uLLIDC

            if "iAmpl" in self._variablesToOptimize:
                qRDotDRhoIAmpl = dv*np.dot(self._qR, self._dRhoIAmpl)
                uLLIAmpl = self._getUpdateDLLIAmpl(qRDotRho=qRDotRho,
                                                    qRDotDRhoIAmpl=
                                                     qRDotDRhoIAmpl,
                                                    cFactorDr=cFactorDr,
                                                    cFactorDRho=cFactorDRho,
                                                    sigma0E=sigma0E,
                                                    y=ys[n],
                                                    r=rs[n],
                                                    dt=dt,
                                                    t=times[n],
                                                    rho=rho)
                self._dLLIAmpls[n] = self._dLLIAmpls[n-1] + uLLIAmpl

            if "iFreq" in self._variablesToOptimize:
                qRDotDRhoIFreq = dv*np.dot(self._qR, self._dRhoIFreq)
                uLLIFreq = self._getUpdateDLLIFreq(qRDotRho=qRDotRho,
                                                    qRDotDRhoIFreq=qRDotDRhoIFreq,
                                                    cFactorDr=cFactorDr,
                                                    cFactorDRho=cFactorDRho,
                                                    sigma0E=sigma0E,
                                                    y=ys[n],
                                                    r=rs[n],
                                                    dt=dt,
                                                    t=times[n],
                                                    rho=rho)
                self._dLLIFreqs[n] = self._dLLIFreqs[n-1] + uLLIFreq

            if "iPhase" in self._variablesToOptimize:
                qRDotDRhoIPhase = dv*np.dot(self._qR, self._dRhoIPhase)
                uLLIPhase = self._getUpdateDLLIPhase(qRDotRho=qRDotRho,
                                                      qRDotDRhoIPhase=qRDotDRhoIPhase,
                                                      cFactorDr=cFactorDr,
                                                      cFactorDRho=cFactorDRho,
                                                      sigma0E=sigma0E,
                                                      y=ys[n],
                                                      r=rs[n],
                                                      dt=dt,
                                                      t=times[n],
                                                      rho=rho)
                self._dLLIPhases[n] = self._dLLIPhases[n-1] + uLLIPhase

        gradients = self._buildGradients(ysSigma=ysSigma,
                                          nTimeSteps=nTimeSteps,
                                          variablesToOptimize=
                                           self._variablesToOptimize)

        return(gradients)

    def _setFixedParameterValues(self, values, variablesToOptimize):
        valuesIndex = 0

        if "leakage" not in variablesToOptimize:
            self._leakage = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "g" not in variablesToOptimize:
            self._g = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "f" not in variablesToOptimize:
            self._f = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "rho0A" not in variablesToOptimize:
            self._rho0A = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "rho0B" not in variablesToOptimize:
            self._rho0B = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eDC" not in variablesToOptimize:
            self._eDC = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eAmpl" not in variablesToOptimize:
            self._eAmpl = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eFreq" not in variablesToOptimize:
            self._eFreq = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "ePhase" not in variablesToOptimize:
            self._ePhase = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iDC" not in variablesToOptimize:
            self._iDC = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iAmpl" not in variablesToOptimize:
            self._iAmpl = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iFreq" not in variablesToOptimize:
            self._iFreq = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iPhase" not in variablesToOptimize:
            self._iPhase = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

    def _setParameterValuesToOptimize(self, values, variablesToOptimize):
        valuesIndex = 0

        if "leakage" in variablesToOptimize:
            self._leakage = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "g" in variablesToOptimize:
            self._g = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "f" in variablesToOptimize:
            self._f = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "rho0A" in variablesToOptimize:
            self._rho0A = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "rho0B" in variablesToOptimize:
            self._rho0B = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eDC" in variablesToOptimize:
            self._eDC = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eAmpl" in variablesToOptimize:
            self._eAmpl = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eFreq" in variablesToOptimize:
            self._eFreq = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "ePhase" in variablesToOptimize:
            self._ePhase = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iDC" in variablesToOptimize:
            self._iDC = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iAmpl" in variablesToOptimize:
            self._iAmpl = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iFreq" in variablesToOptimize:
            self._iFreq = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iPhase" in variablesToOptimize:
            self._iPhase = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

    def _initDerivVariables(self, dt, nTimeSteps, nVSteps, variablesToOptimize,
                                  sigma0E, sigma0I, y0, r0):
        if "leakage" in variablesToOptimize:
            self._dRhoLeakage = np.zeros(nVSteps)
            self._dLLLeakages = np.empty(nTimeSteps)
            self._dLLLeakages[:] = np.nan
            self._dLLLeakages[0] = 0.0

        if "g" in variablesToOptimize:
            self._dRhoG = np.zeros(nVSteps)
            self._dLLGs = np.empty(nTimeSteps)
            self._dLLGs[:] = np.nan
            self._dLLGs[0] = 0.0

        if "f" in variablesToOptimize:
            self._dRhoF = np.zeros(nVSteps)
            self._dLLFs = np.empty(nTimeSteps)
            self._dLLFs[:] = np.nan
            self._dLLFs[0] = 0.0

        if "rho0A" in variablesToOptimize:
            dv = 1.0/nVSteps
            betaRho0Calculator = BetaRho0Calculator(nVSteps=nVSteps)
            rho0 = betaRho0Calculator.getRho0(a=self._rho0A, b=self._rho0B)
            qRDotRho = dv*np.dot(self._qR, rho0)
            cFactorDr = 1-self._g*self._f*qRDotRho
            idM = np.identity(nVSteps)
            cFactorDRho = idM+\
                           dt*(self._leakage/2*self._a0Tilde+\
                                (sigma0E+self._g*self._f*r0)*self._a1-\
                                (sigma0I+self._g*(1-self._f)*r0)*self._a2)
            self._dRhoRho0A = betaRho0Calculator.getDerivRho0(a=self._rho0A,
                                                               b=self._rho0B)[0]
            qRDotDRhoRho0A = dv*np.dot(self._qR, self._dRhoRho0A)
            uLLRho0A = self._getUpdateDLLRho0A(qRDotRho=qRDotRho,
                                                qRDotDRhoRho0A=qRDotDRhoRho0A,
                                                cFactorDr=cFactorDr,
                                                cFactorDRho=cFactorDRho,
                                                sigma0E=sigma0E,
                                                y=y0,
                                                r=r0,
                                                dt=dt,
                                                rho=rho0)
            self._dLLRho0As = np.empty(nTimeSteps)
            self._dLLRho0As[:] = np.nan
            self._dLLRho0As[0] = uLLRho0A

        if "rho0B" in variablesToOptimize:
            dv = 1.0/nVSteps
            betaRho0Calculator = BetaRho0Calculator(nVSteps=nVSteps)
            rho0 = betaRho0Calculator.getRho0(a=self._rho0A, b=self._rho0B)
            qRDotRho = dv*np.dot(self._qR, rho0)
            cFactorDr = 1-self._g*self._f*qRDotRho
            idM = np.identity(nVSteps)
            cFactorDRho = idM+\
                           dt*(self._leakage/2*self._a0Tilde+\
                                (sigma0E+self._g*self._f*r0)*self._a1-\
                                (sigma0I+self._g*(1-self._f)*r0)*self._a2)
            self._dRhoRho0B = betaRho0Calculator.getDerivRho0(a=self._rho0A,
                                                               b=self._rho0B)[1]
            qRDotDRhoRho0B = dv*np.dot(self._qR, self._dRhoRho0B)
            uLLRho0B = self._getUpdateDLLRho0B(qRDotRho=qRDotRho,
                                                qRDotDRhoRho0B=qRDotDRhoRho0B,
                                                cFactorDr=cFactorDr,
                                                cFactorDRho=cFactorDRho,
                                                sigma0E=sigma0E,
                                                y=y0,
                                                r=r0,
                                                dt=dt,
                                                rho=rho0)
            self._dLLRho0Bs = np.empty(nTimeSteps)
            self._dLLRho0Bs[:] = np.nan
            self._dLLRho0Bs[0] = uLLRho0B

        if "eDC" in variablesToOptimize:
            self._dRhoEDC = np.zeros(nVSteps)
            self._dLLEDCs = np.empty(nTimeSteps)
            self._dLLEDCs[:] = np.nan
            self._dLLEDCs[0] = 0.0

        if "eAmpl" in variablesToOptimize:
            self._dRhoEAmpl = np.zeros(nVSteps)
            self._dLLEAmpls = np.empty(nTimeSteps)
            self._dLLEAmpls[:] = np.nan
            self._dLLEAmpls[0] = 0.0

        if "eFreq" in variablesToOptimize:
            self._dRhoEFreq = np.zeros(nVSteps)
            self._dLLEFreqs = np.empty(nTimeSteps)
            self._dLLEFreqs[:] = np.nan
            self._dLLEFreqs[0] = 0.0

        if "ePhase" in variablesToOptimize:
            self._dRhoEPhase = np.zeros(nVSteps)
            self._dLLEPhases = np.empty(nTimeSteps)
            self._dLLEPhases[:] = np.nan
            self._dLLEPhases[0] = 0.0

        if "iDC" in variablesToOptimize:
            self._dRhoIDC = np.zeros(nVSteps)
            self._dLLIDCs = np.empty(nTimeSteps)
            self._dLLIDCs[:] = np.nan
            self._dLLIDCs[0] = 0.0

        if "iAmpl" in variablesToOptimize:
            self._dRhoIAmpl = np.zeros(nVSteps)
            self._dLLIAmpls = np.empty(nTimeSteps)
            self._dLLIAmpls[:] = np.nan
            self._dLLIAmpls[0] = 0.0

        if "iFreq" in variablesToOptimize:
            self._dRhoIFreq = np.zeros(nVSteps)
            self._dLLIFreqs = np.empty(nTimeSteps)
            self._dLLIFreqs[:] = np.nan
            self._dLLIFreqs[0] = 0.0

        if "iPhase" in variablesToOptimize:
            self._dRhoIPhase = np.zeros(nVSteps)
            self._dLLIPhases = np.empty(nTimeSteps)
            self._dLLIPhases[:] = np.nan
            self._dLLIPhases[0] = 0.0

    def _sigma(self, dc, ampl, freq, phase, t):
        sinusoidal = Sinusoidal(dc=dc, ampl=ampl, freq=freq, phase=phase)
        answer = sinusoidal.eval(t=t)
        return(answer)

    def _getUpdateDLLLeakage(self, qRDotDRhoLeakage,
                                   qRDotRho,
                                   cFactorDr,
                                   cFactorDRho,
                                   sigma0E,
                                   y,
                                   r,
                                   dt,
                                   rho):
        drLeakage = sigma0E*(qRDotDRhoLeakage*cFactorDr+\
                             qRDotRho*self._g*self._f*qRDotDRhoLeakage)/\
                     cFactorDr**2
        self._dRhoLeakage = dt*(0.5*self._a0Tilde+self._g*self._f*\
                                drLeakage*self._a1-self._g*(1-self._f)*\
                                drLeakage*self._a2).dot(rho)+\
                             cFactorDRho.dot(self._dRhoLeakage)
        return((y-r)*drLeakage)

    def _getUpdateDLLG(self, qRDotDRhoG,
                             qRDotRho,
                             cFactorDr,
                             cFactorDRho,
                             sigma0E,
                             y,
                             r,
                             dt,
                             rho):
        drG = sigma0E*(qRDotDRhoG*cFactorDr+qRDotRho*\
                        (self._f*qRDotRho+self._g*self._f*qRDotDRhoG))/\
               cFactorDr**2
        self._dRhoG = dt*((self._f*r+self._g*self._f*drG)*self._a1-\
                          ((1-self._f)*r+self._g*self._f*qRDotDRhoG)*\
                          self._a2).dot(rho)+\
                      cFactorDRho.dot(self._dRhoG)
        return((y-r)*drG)

    def _getUpdateDLLF(self, qRDotDRhoF,
                             qRDotRho,
                             cFactorDr,
                             cFactorDRho,
                             sigma0E,
                             y,
                             r,
                             dt,
                             rho):
        drF = sigma0E*(qRDotDRhoF*cFactorDr+\
                        qRDotRho*self._g*(qRDotRho+self._f*qRDotDRhoF))/\
                    cFactorDr**2
        self._dRhoF = dt*(self._g*(r+self._f*drF)*self._a1-\
                           self._g*(-r+(1-self._f)*drF)*self._a2).\
                         dot(rho)+\
                       cFactorDRho.dot(self._dRhoF)
        return((y-r)*drF)

    def _getUpdateDLLRho0A(self, qRDotDRhoRho0A,
                                 qRDotRho,
                                 cFactorDr,
                                 cFactorDRho,
                                 sigma0E,
                                 y,
                                 r,
                                 dt,
                                 rho):
        drRho0A = sigma0E*(qRDotDRhoRho0A*cFactorDr+\
                            qRDotRho*self._g*self._f*qRDotDRhoRho0A)/\
                  cFactorDr**2
        self._dRhoRho0A = dt*(self._g*self._f*drRho0A*self._a1-\
                           self._g*(1-self._f)*drRho0A*self._a2).dot(rho)+\
                       cFactorDRho.dot(self._dRhoRho0A)
        return((y-r)*drRho0A)

    def _getUpdateDLLRho0B(self, qRDotDRhoRho0B,
                                 qRDotRho,
                                 cFactorDr,
                                 cFactorDRho,
                                 sigma0E,
                                 y,
                                 r,
                                 dt,
                                 rho):
        drRho0B = sigma0E*(qRDotDRhoRho0B*cFactorDr+\
                            qRDotRho*self._g*self._f*qRDotDRhoRho0B)/\
                  cFactorDr**2
        self._dRhoRho0B = dt*(self._g*self._f*drRho0B*self._a1-\
                           self._g*(1-self._f)*drRho0B*self._a2).dot(rho)+\
                       cFactorDRho.dot(self._dRhoRho0B)
        return((y-r)*drRho0B)

    def _getUpdateDLLEDC(self, qRDotDRhoEDC,
                               qRDotRho,
                               cFactorDr,
                               cFactorDRho,
                               sigma0E,
                               y,
                               r,
                               dt,
                               t,
                               rho):
        dSigma0EEDC = 1.0
        drEDC, self._dRhoEDC = \
         self._getDRDRhoESigmaParam(dSigma0Param=dSigma0EEDC,
                                     dRhoParam=self._dRhoEDC,
                                     qRDotDRhoParam=qRDotDRhoEDC,
                                     qRDotRho=qRDotRho,
                                     cFactorDr=cFactorDr,
                                     cFactorDRho=cFactorDRho,
                                     sigma0E=sigma0E,
                                     y=y,
                                     r=r,
                                     dt=dt,
                                     t=t,
                                     rho=rho)
        return((y-r)*drEDC)

    def _getUpdateDLLEAmpl(self, qRDotDRhoEAmpl,
                                 qRDotRho,
                                 cFactorDr,
                                 cFactorDRho,
                                 sigma0E,
                                 y,
                                 r,
                                 dt,
                                 t,
                                 rho):
        dSigma0EEAmpl = math.sin(2*math.pi*self._eFreq*t+self._ePhase)
        drEAmpl, self._dRhoEAmpl = \
         self._getDRDRhoESigmaParam(dSigma0Param=dSigma0EEAmpl,
                                     dRhoParam=self._dRhoEAmpl,
                                     qRDotDRhoParam=qRDotDRhoEAmpl,
                                     qRDotRho=qRDotRho,
                                     cFactorDr=cFactorDr,
                                     cFactorDRho=cFactorDRho,
                                     sigma0E=sigma0E,
                                     y=y,
                                     r=r,
                                     dt=dt,
                                     t=t,
                                     rho=rho)
        return((y-r)*drEAmpl)

    def _getUpdateDLLEFreq(self, qRDotDRhoEFreq,
                                 qRDotRho,
                                 cFactorDr,
                                 cFactorDRho,
                                 sigma0E,
                                 y,
                                 r,
                                 dt,
                                 t,
                                 rho):
        dSigma0EEFreq = \
         self._eAmpl*math.cos(2*math.pi*self._eFreq*t+self._ePhase)*2*math.pi*t
        drEFreq, self._dRhoEFreq = \
         self._getDRDRhoESigmaParam(dSigma0Param=dSigma0EEFreq,
                                     dRhoParam=self._dRhoEFreq,
                                     qRDotDRhoParam=qRDotDRhoEFreq,
                                     qRDotRho=qRDotRho,
                                     cFactorDr=cFactorDr,
                                     cFactorDRho=cFactorDRho,
                                     sigma0E=sigma0E,
                                     y=y,
                                     r=r,
                                     dt=dt,
                                     t=t,
                                     rho=rho)
        return((y-r)*drEFreq)

    def _getUpdateDLLEPhase(self, qRDotDRhoEPhase,
                                  qRDotRho,
                                  cFactorDr,
                                  cFactorDRho,
                                  sigma0E,
                                  y,
                                  r,
                                  dt,
                                  t,
                                  rho):
        dSigma0EEPhase = self._eAmpl*\
                          math.cos(2*math.pi*self._eFreq*t+self._ePhase)
        drEPhase, self._dRhoEPhase = \
         self._getDRDRhoESigmaParam(dSigma0Param=dSigma0EEPhase,
                                     dRhoParam=self._dRhoEPhase,
                                     qRDotDRhoParam=qRDotDRhoEPhase,
                                     qRDotRho=qRDotRho,
                                     cFactorDr=cFactorDr,
                                     cFactorDRho=cFactorDRho,
                                     sigma0E=sigma0E,
                                     y=y,
                                     r=r,
                                     dt=dt,
                                     t=t,
                                     rho=rho)
        return((y-r)*drEPhase)

    def _getDRDRhoESigmaParam(self, dSigma0Param,
                                    dRhoParam,
                                    qRDotDRhoParam,
                                    qRDotRho,
                                    cFactorDr,
                                    cFactorDRho,
                                    sigma0E,
                                    y,
                                    r,
                                    dt,
                                    t,
                                    rho):
        drParam = \
         ((dSigma0Param*qRDotRho+sigma0E*qRDotDRhoParam)*cFactorDr+\
          sigma0E*qRDotRho*self._g*self._f*qRDotDRhoParam)/cFactorDr**2
        dRhoParam = \
         dt*((dSigma0Param+self._g*self._f*drParam)*self._a1-\
             self._g*(1-self._f)*drParam*self._a2).dot(rho)+\
         cFactorDRho.dot(dRhoParam)
        return(drParam, dRhoParam)

    def _getUpdateDLLIDC(self, qRDotDRhoIDC,
                               qRDotRho,
                               cFactorDr,
                               cFactorDRho,
                               sigma0E,
                               y,
                               r,
                               dt,
                               t,
                               rho):
        dSigma0IIDC = 1.0
        drIDC, self._dRhoIDC = \
         self._getDRDRhoSigma0IParam(dSigma0IParam=dSigma0IIDC,
                                     dRhoParam=self._dRhoIDC,
                                     qRDotDRhoParam=qRDotDRhoIDC,
                                     qRDotRho=qRDotRho,
                                     cFactorDr=cFactorDr,
                                     cFactorDRho=cFactorDRho,
                                     sigma0E=sigma0E,
                                     y=y,
                                     r=r,
                                     dt=dt,
                                     t=t,
                                     rho=rho)
        return((y-r)*drIDC)

    def _getUpdateDLLIAmpl(self, qRDotDRhoIAmpl,
                                 qRDotRho,
                                 cFactorDr,
                                 cFactorDRho,
                                 sigma0E,
                                 y,
                                 r,
                                 dt,
                                 t,
                                 rho):
        dSigma0IIAmpl = math.sin(2*math.pi*self._iFreq*t+self._iPhase)
        drIAmpl, self._dRhoIAmpl = \
         self._getDRDRhoSigma0IParam(dSigma0IParam=dSigma0IIAmpl,
                                     dRhoParam=self._dRhoIAmpl,
                                     qRDotDRhoParam=qRDotDRhoIAmpl,
                                     qRDotRho=qRDotRho,
                                     cFactorDr=cFactorDr,
                                     cFactorDRho=cFactorDRho,
                                     sigma0E=sigma0E,
                                     y=y,
                                     r=r,
                                     dt=dt,
                                     t=t,
                                     rho=rho)
        return((y-r)*drIAmpl)

    def _getUpdateDLLIFreq(self, qRDotDRhoIFreq,
                                 qRDotRho,
                                 cFactorDr,
                                 cFactorDRho,
                                 sigma0E,
                                 y,
                                 r,
                                 dt,
                                 t,
                                 rho):
        dSigma0IIFreq = self._iAmpl*\
                         math.cos(2*math.pi*self._iFreq*t+self._iPhase)*\
                         2*math.pi*t
        drIFreq, self._dRhoIFreq = \
         self._getDRDRhoSigma0IParam(dSigma0IParam=dSigma0IIFreq,
                                     dRhoParam=self._dRhoIFreq,
                                     qRDotDRhoParam=qRDotDRhoIFreq,
                                     qRDotRho=qRDotRho,
                                     cFactorDr=cFactorDr,
                                     cFactorDRho=cFactorDRho,
                                     sigma0E=sigma0E,
                                     y=y,
                                     r=r,
                                     dt=dt,
                                     t=t,
                                     rho=rho)
        return((y-r)*drIFreq)

    def _getUpdateDLLIPhase(self, qRDotDRhoIPhase,
                                  qRDotRho,
                                  cFactorDr,
                                  cFactorDRho,
                                  sigma0E,
                                  y,
                                  r,
                                  dt,
                                  t,
                                  rho):
        dSigma0IIPhase = self._iAmpl*\
                          math.cos(2*math.pi*self._iFreq*t+self._iPhase)
        drIPhase, self._dRhoIPhase = \
         self._getDRDRhoSigma0IParam(dSigma0IParam=dSigma0IIPhase,
                                      dRhoParam=self._dRhoIPhase,
                                      qRDotDRhoParam=qRDotDRhoIPhase,
                                      qRDotRho=qRDotRho,
                                      cFactorDr=cFactorDr,
                                      cFactorDRho=cFactorDRho,
                                      sigma0E=sigma0E,
                                      y=y,
                                      r=r,
                                      dt=dt,
                                      t=t,
                                      rho=rho)
        return((y-r)*drIPhase)

    def _getDRDRhoSigma0IParam(self, dSigma0IParam,
                                     qRDotDRhoParam,
                                     dRhoParam,
                                     qRDotRho,
                                     cFactorDr,
                                     cFactorDRho,
                                     sigma0E,
                                     y,
                                     r,
                                     dt,
                                     t,
                                     rho):
        drParam = \
         sigma0E*(qRDotDRhoParam*cFactorDr+
                   qRDotRho*self._g*self._f*qRDotDRhoParam)/cFactorDr**2
        dRhoParam = dt*(self._g*self._f*drParam*self._a1-\
                         (1.0+self._g*(1-self._f))*dSigma0IParam*self._a2).\
                         dot(rho)+cFactorDRho.dot(dRhoParam)
        return(drParam, dRhoParam)

    def _buildGradients(self, ysSigma, nTimeSteps, variablesToOptimize):
        gradients = []
        normalization = np.arange(1, nTimeSteps+1)*ysSigma**2

        if "leakage" in variablesToOptimize:
            gradients.append(self._dLLLeakages/normalization)

        if "g" in variablesToOptimize:
            gradients.append(self._dLLGs/normalization)

        if "f" in variablesToOptimize:
            gradients.append(self._dLLFs/normalization)

        if "rho0A" in variablesToOptimize:
            gradients.append(self._dLLRho0As/normalization)

        if "rho0B" in variablesToOptimize:
            gradients.append(self._dLLRho0Bs/normalization)

        if "eDC" in variablesToOptimize:
            gradients.append(self._dLLEDCs/normalization)

        if "eAmpl" in variablesToOptimize:
            gradients.append(self._dLLEAmpls/normalization)

        if "eFreq" in variablesToOptimize:
            gradients.append(self._dLLEFreqs/normalization)

        if "ePhase" in variablesToOptimize:
            gradients.append(self._dLLEPhases/normalization)

        if "iDC" in variablesToOptimize:
            gradients.append(self._dLLIDCs/normalization)

        if "iAmpl" in variablesToOptimize:
            gradients.append(self._dLLIAmpls/normalization)

        if "iFreq" in variablesToOptimize:
            gradients.append(self._dLLIFreqs/normalization)

        if "iPhase" in variablesToOptimize:
            gradients.append(self._dLLIPhases/normalization)

        return(gradients)
