
import sys
import pdb
import math
import numpy as np
from utilsMath import Logistic

# Begin delete
# import matplotlib.pyplot as plt
# End delete

class IFEDMWithInputLayerGradientsCalculator:

    def __init__(self, a0Tilde, a1, a2, reversedQs, eFilterSize, iFilterSize,
                       variablesToOptimize, fixedParameterValues):
        self._a0Tilde = a0Tilde
        self._a1 = a1
        self._a2 = a2
        self._qR = reversedQs
        self._eFSize = eFilterSize
        self._iFSize = iFilterSize
        self._variablesToOptimize = variablesToOptimize
        self._setFixedParameterValues(values=fixedParameterValues,
                                       variablesToOptimize=variablesToOptimize)

    def deriv(self, x, ys, rs, rhos, inputs, ysSigma, dt):
        self._setParameterValuesToOptimize(values=x, 
                                            variablesToOptimize=\
                                             self._variablesToOptimize)

        nTimeSteps = ys.size
        nVSteps = self._a0Tilde.shape[0]
        dv = 1.0/nVSteps
        idM = np.identity(self._a0Tilde.shape[0])
        self._initDerivVariables(nVSteps=nVSteps, 
                                  nTimeSteps=nTimeSteps,
                                  variablesToOptimize=self._variablesToOptimize)

        for n in xrange(1, nTimeSteps):
            '''
            Derivatives of r are computed for time n
            Derivatives of rho are computed for time n+1
            '''
            # '''
            if n%100==0:
                print("Gradient calculator step %d (out of %d)" % \
                      (n, nTimeSteps))
                if "leakage" in self._variablesToOptimize:
                    print "DLLLeakage=%f, "%self._dLLLeakages[n-1],
                if "g" in self._variablesToOptimize:
                    print "DLLG=%f, "%self._dLLGs[n-1],
                if "f" in self._variablesToOptimize:
                    print "DLLF=%f, "%self._dLLFs[n-1],
                if "eL" in self._variablesToOptimize:
                    print "DLLEL=%f, "%self._dLLELs[n-1],
                if "eK" in self._variablesToOptimize:
                    print "DLLEK=%f, "%self._dLLEKs[n-1],
                if "eX0" in self._variablesToOptimize:
                    print "DLLEX0=%f, "%self._dLLEX0s[n-1],
                if "iL" in self._variablesToOptimize:
                    print "DLLIL=%f, "%self._dLLILs[n-1],
                if "iK" in self._variablesToOptimize:
                    print "DLLIK=%f, "%self._dLLIKs[n-1],
                if "iX0" in self._variablesToOptimize:
                    print "DLLIX0=%f, "%self._dLLIX0s[n-1],
                if "eF" in self._variablesToOptimize:
                    print "DLLEF=%s, "%str(self._dLLEFs[:,n-1]),
                if "iF" in self._variablesToOptimize:
                    print "DLLIF=%s, "%str(self._dLLIFs[:,n-1]),
                print
                sys.stdout.flush()
            # '''

            rhoAtN = rhos[:,n]
            rAtN = rs[n]
            sAtN = inputs[n,:]

            sigma0E = self._sigma(s=sAtN, filter=self._eF, l=self._eL,
                                          k=self._eK, x0=self._eX0)
            sigma0I = self._sigma(s=sAtN, filter=self._iF, l=self._iL,
                                          k=self._iK, x0=self._iX0)

            qRDotRho = dv*np.dot(self._qR, rhoAtN)
            cFactorDr = 1-self._g*self._f*qRDotRho
            cFactorDRho = idM+\
                           dt*(self._leakage/2*self._a0Tilde+\
                                      (sigma0E+self._g*self._f*rAtN)*self._a1-\
                                      (sigma0I+self._g*(1-self._f)*rAtN)*\
                                      self._a2)

            if "leakage" in self._variablesToOptimize:
                qRDotDRhoLeakage = dv*np.dot(self._qR, self._dRhoLeakage)
                uLLLeakage = self._getUpdateDLLLeakage(qRDotRho=qRDotRho, 
                                                        qRDotDRhoLeakage=
                                                         qRDotDRhoLeakage,
                                                        cFactorDr=cFactorDr,
                                                        cFactorDRho=cFactorDRho,
                                                        sigma0E=sigma0E,
                                                        y=ys[n],
                                                        r=rs[n],
                                                        dt=dt,
                                                        rhoAtN=rhoAtN)
                self._dLLLeakages[n] = self._dLLLeakages[n-1] + uLLLeakage

            if "g" in self._variablesToOptimize:
                qRDotDRhoG = dv*np.dot(self._qR, self._dRhoG)
                uLLG = self._getUpdateDLLG(qRDotRho=qRDotRho,
                                            qRDotDRhoG=qRDotDRhoG,
                                            cFactorDr=cFactorDr, 
                                            cFactorDRho=cFactorDRho,
                                            sigma0E=sigma0E,
                                            y=ys[n], 
                                            r=rs[n],
                                            dt=dt,
                                            rhoAtN=rhoAtN,
                                            rAtN=rAtN)
                self._dLLGs[n] = self._dLLGs[n-1] + uLLG

            if "f" in self._variablesToOptimize:
                qRDotDRhoF = dv*np.dot(self._qR, self._dRhoF)
                uLLF = self._getUpdateDLLF(qRDotRho=qRDotRho,
                                            qRDotDRhoF=qRDotDRhoF,
                                            cFactorDr=cFactorDr, 
                                            cFactorDRho=cFactorDRho,
                                            sigma0E=sigma0E,
                                            y=ys[n], 
                                            r=rs[n],
                                            dt=dt,
                                            rhoAtN=rhoAtN,
                                            rAtN=rAtN)
                self._dLLFs[n] = self._dLLFs[n-1] + uLLF

            if "eL" in self._variablesToOptimize or \
               "eK" in self._variablesToOptimize or \
               "eX0" in self._variablesToOptimize:
                sDotEF = np.dot(sAtN, self._eF)
            if "iL" in self._variablesToOptimize or \
               "iK" in self._variablesToOptimize or \
               "iX0" in self._variablesToOptimize:
                sDotIF = np.dot(sAtN, self._iF)

            if "eL" in self._variablesToOptimize:
                qRDotDRhoEL = dv*np.dot(self._qR, self._dRhoEL)
                uLLEL = self._getUpdateDLLEL(qRDotRho=qRDotRho,
                                              qRDotDRhoEL=qRDotDRhoEL,
                                              cFactorDr=cFactorDr, 
                                              cFactorDRho=cFactorDRho,
                                              sigma0E=sigma0E,
                                              sDotEF=sDotEF,
                                              y=ys[n], 
                                              r=rs[n],
                                              dt=dt,
                                              rhoAtN=rhoAtN)
                self._dLLELs[n] = self._dLLELs[n-1] + uLLEL

            if "eK" in self._variablesToOptimize:
                qRDotDRhoEK = dv*np.dot(self._qR, self._dRhoEK)
                uLLEK = self._getUpdateDLLEK(qRDotRho=qRDotRho,
                                              qRDotDRhoEK=qRDotDRhoEK,
                                              cFactorDr=cFactorDr, 
                                              cFactorDRho=cFactorDRho,
                                              sigma0E=sigma0E,
                                              sDotEF=sDotEF,
                                              y=ys[n], 
                                              r=rs[n],
                                              dt=dt,
                                              rhoAtN=rhoAtN)
                self._dLLEKs[n] = self._dLLEKs[n-1] + uLLEL

            if "eX0" in self._variablesToOptimize:
                qRDotDRhoEX0 = dv*np.dot(self._qR, self._dRhoEX0)
                uLLEX0 = self._getUpdateDLLEX0(qRDotRho=qRDotRho,
                                                qRDotDRhoEX0=qRDotDRhoEX0,
                                                cFactorDr=cFactorDr, 
                                                cFactorDRho=cFactorDRho,
                                                sigma0E=sigma0E,
                                                sDotEF=sDotEF,
                                                y=ys[n], 
                                                r=rs[n],
                                                dt=dt,
                                                rhoAtN=rhoAtN)
                self._dLLEX0s[n] = self._dLLEX0s[n-1] + uLLEX0

            if "iL" in self._variablesToOptimize:
                qRDotDRhoIL = dv*np.dot(self._qR, self._dRhoIL)
                uDLLIL = self._getUpdateDLLIL(qRDotRho=qRDotRho,
                                               qRDotDRhoIL=qRDotDRhoIL,
                                               cFactorDr=cFactorDr, 
                                               cFactorDRho=cFactorDRho,
                                               sigma0E=sigma0E,
                                               sDotIF=sDotIF,
                                               y=ys[n], 
                                               r=rs[n],
                                               dt=dt,
                                               rhoAtN=rhoAtN)
                self._dLLILs[n] = self._dLLILs[n-1] + uDLLIL

            if "iK" in self._variablesToOptimize:
                qRDotDRhoIK = dv*np.dot(self._qR, self._dRhoIK)
                uDLLIK = self._getUpdateDLLIK(qRDotRho=qRDotRho,
                                               qRDotDRhoIK=qRDotDRhoIK,
                                               cFactorDr=cFactorDr, 
                                               cFactorDRho=cFactorDRho,
                                               sigma0E=sigma0E,
                                               sDotIF=sDotIF,
                                               y=ys[n], 
                                               r=rs[n],
                                               dt=dt,
                                               rhoAtN=rhoAtN)
                self._dLLIKs[n] = self._dLLIKs[n-1] + uDLLIK

            if "iX0" in self._variablesToOptimize:
                qRDotDRhoIX0 = dv*np.dot(self._qR, self._dRhoIX0)
                uDLLIX0 = self._getUpdateDLLIX0(qRDotRho=qRDotRho,
                                                 qRDotDRhoIX0=qRDotDRhoIX0,
                                                 cFactorDr=cFactorDr, 
                                                 cFactorDRho=cFactorDRho,
                                                 sigma0E=sigma0E,
                                                 sDotIF=sDotIF,
                                                 y=ys[n], 
                                                 r=rs[n],
                                                 dt=dt,
                                                 rhoAtN=rhoAtN)
                self._dLLIX0s[n] = self._dLLIX0s[n-1] + uDLLIX0

            if "eF" in self._variablesToOptimize:
                qRDotDRhoEF = dv*np.dot(self._qR, self._dRhoEF)
                uDLLEF = self._getUpdateDLLEF(qRDotRho=qRDotRho,
                                              qRDotDRhoEF=qRDotDRhoEF,
                                              cFactorDr=cFactorDr, 
                                              cFactorDRho=cFactorDRho,
                                              sigma0E=sigma0E,
                                              sAtN=sAtN,
                                              y=ys[n], 
                                              r=rs[n],
                                              dt=dt,
                                              rhoAtN=rhoAtN)
                self._dLLEFs[:,n] = self._dLLEFs[:,n-1] + uDLLEF

            if "iF" in self._variablesToOptimize:
                qRDotDRhoIF=dv*np.dot(self._qR, self._dRhoIF)
                uDLLIF = self._getUpdateDLLIF(qRDotRho=qRDotRho,
                                               qRDotDRhoIF=qRDotDRhoIF,
                                               cFactorDr=cFactorDr, 
                                               cFactorDRho=cFactorDRho,
                                               sigma0E=sigma0E,
                                               sigma0I=sigma0I,
                                               sAtN=sAtN,
                                               y=ys[n], 
                                               r=rs[n],
                                               dt=dt,
                                               rhoAtN=rhoAtN)
                self._dLLIFs[:,n] = self._dLLIFs[:,n-1] + uDLLIF

        gradients = self._buildGradients(ysSigma=ysSigma,
                                          nTimeSteps=nTimeSteps,
                                          variablesToOptimize=
                                           self._variablesToOptimize)

        # Begin delete
        # plt.close("all")
        # plt.plot(self._dLLLeakages, label="dLLLeakage", color="red")
        # plt.ylabel("dLLLeakage")
        # plt.xlabel("Time (ms)")
        # # ax2 = plt.twinx()
        # # ax2.plot(inputs, label="Input", color="blue")
        # # ax2.set_ylabel("Input")
        # plt.grid()
        # plt.legend()
        # plt.title("leakage=%.02f"%(x[0]))
        # plt.show()
        # pdb.set_trace()
        # End delete

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

        if "eL" not in variablesToOptimize:
            self._eL = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eK" not in variablesToOptimize:
            self._eK = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eX0" not in variablesToOptimize:
            self._eX0 = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iL" not in variablesToOptimize:
            self._iL = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iK" not in variablesToOptimize:
            self._iK = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iX0" not in variablesToOptimize:
            self._iX0 = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eF" not in variablesToOptimize:
            self._eF = np.array(values[valuesIndex:(valuesIndex+self._eFSize)])
            valuesIndex =  valuesIndex + self._eFSize

        if "iF" not in variablesToOptimize:
            self._iF = np.array(values[valuesIndex:(valuesIndex+self._iFSize)])
            valuesIndex =  valuesIndex + self._iFSize

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

        if "eL" in variablesToOptimize:
            self._eL = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eK" in variablesToOptimize:
            self._eK = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eX0" in variablesToOptimize:
            self._eX0 = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iL" in variablesToOptimize:
            self._iL = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iK" in variablesToOptimize:
            self._iK = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "iX0" in variablesToOptimize:
            self._iX0 = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eF" in variablesToOptimize:
            self._eF = np.array(values[valuesIndex:(valuesIndex+self._eFSize)])
            valuesIndex =  valuesIndex + self._eFSize

        if "iF" in variablesToOptimize:
            self._iF = np.array(values[valuesIndex:(valuesIndex+self._iFSize)])
            valuesIndex =  valuesIndex + self._iFSize

    def _initDerivVariables(self, nTimeSteps, nVSteps, variablesToOptimize):
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
        if "eL" in variablesToOptimize:
            self._dRhoEL = np.zeros(nVSteps)
            self._dLLELs = np.empty(nTimeSteps)
            self._dLLELs[:] = np.nan
            self._dLLELs[0] = 0.0
        if "eK" in variablesToOptimize:
            self._dRhoEK = np.zeros(nVSteps)
            self._dLLEKs = np.empty(nTimeSteps)
            self._dLLEKs[:] = np.nan
            self._dLLEKs[0] = 0.0
        if "eX0" in variablesToOptimize:
            self._dRhoEX0 = np.zeros(nVSteps)
            self._dLLEX0s = np.empty(nTimeSteps)
            self._dLLEX0s[:] = np.nan
            self._dLLEX0s[0] = 0.0
        if "iL" in variablesToOptimize:
            self._dRhoIL = np.zeros(nVSteps)
            self._dLLILs = np.empty(nTimeSteps)
            self._dLLILs[:] = np.nan
            self._dLLILs[0] = 0.0
        if "iK" in variablesToOptimize:
            self._dRhoIK = np.zeros(nVSteps)
            self._dLLIKs = np.empty(nTimeSteps)
            self._dLLIKs[:] = np.nan
            self._dLLIKs[0] = 0.0
        if "iX0" in variablesToOptimize:
            self._dRhoIX0 = np.zeros(nVSteps)
            self._dLLIX0s = np.empty(nTimeSteps)
            self._dLLIX0s[:] = np.nan
            self._dLLIX0s[0] = 0.0
        if "eF" in variablesToOptimize:
            self._dRhoEF = np.zeros((nVSteps, self._eF.size))
            self._dLLEFs = np.empty((self._eF.size, nTimeSteps))
            self._dLLEFs[:] = np.nan
            self._dLLEFs[:,0] = np.zeros(self._eF.size)
        if "iF" in variablesToOptimize:
            self._dRhoIF = np.zeros((nVSteps, self._iF.size))
            self._dLLIFs = np.empty((self._iF.size, nTimeSteps))
            self._dLLIFs[:] = np.nan
            self._dLLIFs[:,0] = np.zeros(self._iF.size)

    def _sigma(self, s, filter, l, k, x0):
        rectification = Logistic(k=k, x0=x0, l=l)
        return(rectification.eval(x=np.dot(s, filter)))


    def _getUpdateDLLLeakage(self, qRDotRho,
                                   qRDotDRhoLeakage,
                                   cFactorDr,
                                   cFactorDRho,
                                   sigma0E,
                                   y,
                                   r,
                                   dt,
                                   rhoAtN):
        drLeakage = sigma0E*(qRDotDRhoLeakage*cFactorDr+\
                                   qRDotRho*self._g*self._f*qRDotDRhoLeakage)/\
                           cFactorDr**2
        self._dRhoLeakage = dt*(0.5*self._a0Tilde+self._g*self._f*\
                                drLeakage*self._a1-self._g*(1-self._f)*\
                                drLeakage*self._a2).dot(rhoAtN)+\
                             cFactorDRho.dot(self._dRhoLeakage)
        return((y-r)*drLeakage)

    def _getUpdateDLLG(self, qRDotRho,
                             qRDotDRhoG,
                             cFactorDr,
                             cFactorDRho,
                             sigma0E,
                             y,
                             r,
                             dt,
                             rhoAtN,
                             rAtN):
        drG = sigma0E*(qRDotDRhoG*cFactorDr+qRDotRho*\
                        (self._f*qRDotRho+self._g*self._f*qRDotDRhoG))/\
               cFactorDr**2
        self._dRhoG = dt*((self._f*rAtN+self._g*self._f*drG)*self._a1-\
                          ((1-self._f)*rAtN+self._g*self._f*qRDotDRhoG)*\
                          self._a2).dot(rhoAtN)+\
                      cFactorDRho.dot(self._dRhoG)
        return((y-r)*drG)

    def _getUpdateDLLF(self, qRDotRho,
                             qRDotDRhoF,
                             cFactorDr,
                             cFactorDRho,
                             sigma0E,
                             y,
                             r,
                             dt,
                             rhoAtN,
                             rAtN):
        drF = sigma0E*(qRDotDRhoF*cFactorDr+\
                        qRDotRho*self._g*(qRDotRho+self._f*qRDotDRhoF))/\
                    cFactorDr**2
        self._dRhoF = dt*(self._g*(rAtN+self._f*drF)*self._a1-\
                           self._g*(-rAtN+(1-self._f)*drF)*self._a2).\
                         dot(rhoAtN)+\
                       cFactorDRho.dot(self._dRhoF)
        return((y-r)*drF)

    def _getUpdateDLLEL(self, qRDotRho,
                              qRDotDRhoEL,
                              cFactorDr,
                              cFactorDRho,
                              sigma0E,
                              sDotEF,
                              y,
                              r,
                              dt,
                              rhoAtN):
        dSigma0EEL = 1.0/(1+math.exp(-self._eK*(sDotEF-self._eX0)))
        drEL = ((dSigma0EEL*qRDotRho+sigma0E*qRDotDRhoEL)*cFactorDr+\
                sigma0E*qRDotRho*self._g*self._f*qRDotDRhoEL)/cFactorDr**2
        self._dRhoEL = dt*((dSigma0EEL+self._g*self._f*drEL)*self._a1-\
                           self._g*(1-self._f)*drEL*self._a2).dot(rhoAtN)+\
                         cFactorDRho.dot(self._dRhoEL)
        return((y-r)*drEL)

    def _getUpdateDLLEK(self, qRDotRho,
                              qRDotDRhoEK,
                              cFactorDr,
                              cFactorDRho,
                              sigma0E,
                              sDotEF,
                              y,
                              r,
                              dt,
                              rhoAtN):
        dSigma0EEK = self._eL*math.exp(-self._eK*\
                                       (sDotEF-self._eX0))*\
                      (sDotEF-self._eX0)/\
                      (1+math.exp(-self._eK*(sDotEF-self._eX0)))**2
        drEK = ((dSigma0EEK*qRDotRho+sigma0E*qRDotDRhoEK)*\
                       cFactorDr+sigma0E*qRDotRho*(self._g*self._f*qRDotDRhoEK))/\
                       cFactorDr**2
        self._dRhoEK = dt*((dSigma0EEK+self._g*self._f*drEK)*self._a1-\
                           self._g*(1-self._f)*drEK*self._a2).dot(rhoAtN)+\
                       cFactorDRho.dot(self._dRhoEK)
        return((y-r)*drEK)

    def _getUpdateDLLEX0(self, qRDotRho,
                               qRDotDRhoEX0,
                               cFactorDr,
                               cFactorDRho,
                               sigma0E,
                               sDotEF,
                               y,
                               r,
                               dt,
                               rhoAtN):
        dSigma0EEX0 = -self._eL*math.exp(-self._eK*(sDotEF-self._eX0)*\
                                         self._eK)/\
                       (1+math.exp(-self._eK*(sDotEF-self._eX0)))**2
        drEX0 = ((dSigma0EEX0*qRDotRho+sigma0E*qRDotDRhoEX0)*cFactorDr+\
                 sigma0E*qRDotRho*(self._g*self._f*qRDotDRhoEX0))/cFactorDr**2
        self._dRhoEX0 = dt*((dSigma0EEX0+self._g*self._f*drEX0)*self._a1-\
                            self._g*(1-self._f)*drEX0*self._a2).dot(rhoAtN)+\
                        cFactorDRho.dot(self._dRhoEX0)
        return((y-r)*drEX0)

    def _getUpdateDLLIL(self, qRDotRho,
                              qRDotDRhoIL,
                              cFactorDr,
                              cFactorDRho,
                              sigma0E,
                              sDotIF,
                              y,
                              r,
                              dt,
                              rhoAtN):
        dSigma0IIL = 1.0/(1+math.exp(-self._iK*(sDotIF-self._iX0)))
        drIL = sigma0E*(qRDotDRhoIL*cFactorDr+qRDotRho*self._g*\
                         self._f*qRDotDRhoIL)/cFactorDr**2
        self._dRhoIL = dt*(self._g*self._f*drIL*self._a1-\
                        (dSigma0IIL+self._g*(1-self._f)*drIL)*self._a2).dot(rhoAtN)+\
                        cFactorDRho.dot(self._dRhoIL)
        return((y-r)*drIL)

    def _getUpdateDLLIK(self, qRDotRho,
                              qRDotDRhoIK,
                              cFactorDr,
                              cFactorDRho,
                              sigma0E,
                              sDotIF,
                              y,
                              r,
                              dt,
                              rhoAtN):
        dSigma0IIK = self._iL*math.exp(-self._iK*(sDotIF-self._iX0))*\
                      (sDotIF-self._iX0)/\
                      (1+math.exp(-self._iK*(sDotIF-self._iX0)))**2
        drIK = sigma0E*(qRDotDRhoIK*cFactorDr+qRDotRho*self._g*\
                               self._f*qRDotDRhoIK)/\
                      cFactorDr**2
        self._dRhoIK = dt*(self._g*self._f*drIK*self._a1-\
                           (dSigma0IIK+self._g*(1-self._f)*drIK)*self._a2).\
                           dot(rhoAtN)+\
                        cFactorDRho.dot(self._dRhoIK)
        return((y-r)*drIK)

    def _getUpdateDLLIX0(self, qRDotRho,
                               qRDotDRhoIX0,
                               cFactorDr,
                               cFactorDRho,
                               sigma0E,
                               sDotIF,
                               y,
                               r,
                               dt,
                               rhoAtN):
        dSigma0IIX0 = -self._iL*math.exp(-self._iK*(sDotIF-self._iX0)*\
                                         self._iK)/\
                              (1+math.exp(-self._iK*(sDotIF-self._iX0)))**2
        drIX0 = sigma0E*(qRDotDRhoIX0*cFactorDr+\
                               qRDotRho*self._g*self._f*qRDotDRhoIX0)/\
                      cFactorDr**2
        self._dRhoIX0 = dt*(self._g*self._f*drIX0*self._a1-\
                            (dSigma0IIX0+self._g*(1-self._f)*drIX0)*\
                            self._a2).dot(rhoAtN)+\
                         cFactorDRho.dot(self._dRhoIX0)
        return((y-r)*drIX0)

    def _getUpdateDLLEF(self, qRDotRho,
                              qRDotDRhoEF,
                              cFactorDr,
                              cFactorDRho,
                              sigma0E,
                              sAtN,
                              y,
                              r,
                              dt,
                              rhoAtN):
        dSigma0EEF = self._eK*sigma0E*(1-sigma0E/self._eL)*sAtN
        drEF = (dSigma0EEF*qRDotRho+sigma0E*qRDotDRhoEF*cFactorDr+\
                sigma0E*qRDotRho*self._g*self._f*qRDotDRhoEF)/cFactorDr**2
        self._dRhoEF = dt*(np.outer(self._a1.dot(rhoAtN), 
                                    dSigma0EEF+self._g*self._f*drEF)-\
                           np.outer(self._a2.dot(rhoAtN), self._g*(1-self._f)*\
                                                       drEF))+\
                       cFactorDRho.dot(self._dRhoEF)
        return((y-r)*drEF)

    def _getUpdateDLLIF(self, qRDotRho,
                              qRDotDRhoIF,
                              cFactorDr,
                              cFactorDRho,
                              sigma0E,
                              sigma0I,
                              sAtN,
                              y,
                              r,
                              dt,
                              rhoAtN):
        dSigma0IIF = self._iK*sigma0I*(1-sigma0I/self._iL)*sAtN
        drIF = (sigma0E*qRDotDRhoIF*cFactorDr+\
                sigma0E*qRDotRho*self._g*self._f*qRDotDRhoIF)/cFactorDr**2
        self._dRhoIF = dt*(self._g*self._f*np.outer(self._a1.dot(rhoAtN), drIF)-\
                           np.outer(self._a2.dot(rhoAtN), 
                                     dSigma0IIF+self._g*(1-self._f)*drIF))+\
                        cFactorDRho.dot(self._dRhoIF)
        return((y-r)*drIF)

    def _buildGradients(self, ysSigma, nTimeSteps, variablesToOptimize):
        gradients = []
        normalization = np.arange(1, nTimeSteps+1)*ysSigma**2

        if "leakage" in variablesToOptimize:
            gradients.append(self._dLLLeakages/normalization)

        if "g" in variablesToOptimize:
            gradients.append(self._dLLGs/normalization)

        if "f" in variablesToOptimize:
            gradients.append(self._dLLFs/normalization)

        if "eL" in variablesToOptimize:
            gradients.append(self._dLLELs/normalization)

        if "eK" in variablesToOptimize:
            gradients.append(self._dLLEKs/normalization)

        if "eX0" in variablesToOptimize:
            gradients.append(self._dLLEX0s/normalization)

        if "iL" in variablesToOptimize:
            gradients.append(self._dLLILs/normalization)

        if "iK" in variablesToOptimize:
            gradients.append(self._dLLIKs/normalization)

        if "iX0" in variablesToOptimize:
            gradients.append(self._dLLIX0s/normalization)

        if "eF" in variablesToOptimize:
            for i in xrange(self._dLLEFs.shape[0]):
                gradients.append(self._dLLEFs[i,:]/normalization)

        if "iF" in variablesToOptimize:
            for i in xrange(self._dLLIFs.shape[0]):
                gradients.append(self._dLLIFs[i,:]/normalization)

        return(gradients)

