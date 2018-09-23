
import sys
import pdb
import math
import numpy as np
from edmsMath import Logistic

class IFEDMWithInputLayerGradientCalculator:

    def __init__(self, a0Tilde, a1, a2, dt, reversedQs, stimuliSource):
        self._a0Tilde = a0Tilde
        self._a1 = a1
        self._a2 = a2
        self._dt = dt
        self._qR = reversedQs
        self._stimuliSource = stimuliSource

    def _sigma(self, s, filter, l, k, x0):
        rectification = Logistic(k=k, x0=x0, l=l)
        return(rectification.eval(x=np.dot(s, filter)))

    def deriv(self, times, ys, rs, rhos, leakage, g, f, 
                    eL, eK, eX0, 
                    iL, iK, iX0, 
                    eF, iF):
        # y?s and r?s should be sampled at dtSaveRhos
        dRhoLeakage = dRhoG = dRhoF = dRhoEL = dRhoEK = dRhoEX0 = \
         dRhoIL = dRhoIK = dRhoIX0 = np.zeros(self._a0Tilde.shape[0])
        dRhoEF = dRhoIF = np.zeros((self._a0Tilde.shape[0], eF.size))

        drLeakage = drG = drF = drEL = drEK = drEX0 = drIL = drIK = drIX0 =0
        drEF = drIF = np.zeros(eF.size)
        idM = np.identity(self._a0Tilde.shape[0])
        dLLLeakage = dLLG = dLLF = dLLEL = dLLEK = dLLEX0 = \
         dLLIL = dLLIK = dLLIX0 = 0
        dLLEF = dLLIF = np.zeros(eF.size)

        drLeakages = np.empty(ys.size)
        drGs = np.empty(ys.size)
        drFs = np.empty(ys.size)
        drELs = np.empty(ys.size)
        drEKs = np.empty(ys.size)
        drEX0s = np.empty(ys.size)
        drILs = np.empty(ys.size)
        drIKs = np.empty(ys.size)
        drIX0s = np.empty(ys.size)
        drEFs = np.empty((eF.size, ys.size))
        drIFs = np.empty((eF.size, ys.size))

        drLeakages[0] = drLeakage
        drGs[0] = drG
        drFs[0] = drF
        drELs[0] = drEL
        drEKs[0] = drEK
        drEX0s[0] = drEX0
        drILs[0] = drIL
        drIKs[0] = drIK
        drIX0s[0] = drIX0
        drEFs[:, 0] = drEF
        drIFs[:, 0] = drIF
        for n in xrange(times.size):
            '''
            Derivatives of r are computed for time n
            Derivatives of rho are computed for time n+1
            '''
            if n%100==0:
                print("Processing time %.05f out of %.02f" % \
                      (times[n], times[-1]))
                sys.stdout.flush()

            rhoAtN = rhos[:,n]
            rAtN = rs[n]
            sAtN = self._stimuliSource.getStimulus(t=times[n])

            sigma0E = self._sigma(s=sAtN, filter=eF, l=eL, k=eK, x0=eX0)
            sigma0I = self._sigma(s=sAtN, filter=iF, l=iL, k=iK, x0=iX0)
            qRDotRho = np.dot(self._qR, rhos[:,n])
            cFactorDr = 1-g*f*qRDotRho
            cFactorDRho = idM+\
                           self._dt*(leakage/2*self._a0Tilde+\
                                      (sigma0E+g*f*rAtN)*self._a1-\
                                      (sigma0I+g*(1-f)*rAtN)*self._a2)

            # Begin derivative wrt leakage
            qRDotDRhoLeakage = np.dot(self._qR, dRhoLeakage)
            drLeakage = sigma0E*(qRDotDRhoLeakage*cFactorDr+\
                                  qRDotRho*g*f*qRDotDRhoLeakage)/cFactorDr**2
            dRhoLeakage = self._dt*(0.5*self._a0Tilde+g*f*drLeakage*self._a1-\
                                   g*(1-f)*drLeakage*self._a2).dot(rhoAtN)+\
                          cFactorDRho.dot(dRhoLeakage)
            # End derivative wrt leakage

            # Begin derivative wrt g
            qRDotDRhoG = np.dot(self._qR, dRhoG)
            drG = (sigma0E*qRDotDRhoG*cFactorDr+\
                               sigma0E*qRDotRho*(f*qRDotRho+g*f*qRDotDRhoG))/\
                   cFactorDr**2
            dRhoG = self._dt*((f*rAtN+g*f*drG)*self._a1-\
                                   ((1-f)*rAtN+g*f*qRDotDRhoG)*self._a2).dot(rhoAtN)+\
                          cFactorDRho.dot(dRhoG)
            # End derivative wrt g

            # Begin derivative wrt f
            qRDotDRhoF = np.dot(self._qR, dRhoF)
            drF = (sigma0E*qRDotDRhoF*cFactorDr+\
                               sigma0E*qRDotRho*(g*qRDotRho+g*f*qRDotDRhoF))/\
                   cFactorDr**2
            dRhoF = self._dt*(g*(rAtN*(drF))*self._a1-g*(-rAtN+(1-f)*drF)*self._a2).dot(rhoAtN)+\
                     cFactorDRho.dot(dRhoF)
            # End derivative wrt f

            # Begin precalculation for derivatives wrt sigmoidal parameters
            sDotEF = np.dot(sAtN, eF)
            dSigma0EEL = 1.0/(1+math.exp(-eK*(sDotEF-eX0)))
            dSigma0EEK = eL*math.exp(-eK*(sDotEF-eX0))*(sDotEF-eX0)/(1+math.exp(-eK*(sDotEF-eX0)))**2
            dSigma0EEX0 = -eL*math.exp(-eK*(sDotEF-eX0)*eK)/(1+math.exp(-eK*(sDotEF-eX0)))**2

            sDotIF = np.dot(sAtN, iF)
            dSigma0IIL = 1.0/(1+math.exp(-iK*(sDotIF-iX0)))
            dSigma0IIK = iL*math.exp(-iK*(sDotIF-iX0))*(sDotIF-iX0)/(1+math.exp(-iK*(sDotIF-iX0)))**2
            dSigma0IIX0 = -iL*math.exp(-iK*(sDotIF-iX0)*iK)/(1+math.exp(-iK*(sDotIF-iX0)))**2
            # End precalculation for derivatives wrt sigmoidal parameters

            # Begin derivative wrt EL
            qRDotDRhoEL = np.dot(self._qR, dRhoEL)
            drEL = ((dSigma0EEL*qRDotRho+sigma0E*qRDotDRhoEL)*cFactorDr+\
                    sigma0E*qRDotRho*g*f*qRDotDRhoEL)/cFactorDr**2
            dRhoEL = self._dt*((dSigma0EEL+g*f*drEL)*self._a1-\
                              g*(1-f)*drEL*self._a2).dot(rhoAtN)+\
                     cFactorDRho.dot(dRhoEL)
            # End derivative wrt EL

            # Begin derivative wrt eK
            qRDotDRhoEK = np.dot(self._qR, dRhoEK)
            drEK = ((dSigma0EEK*qRDotRho+sigma0E*qRDotDRhoEK)*cFactorDr+\
                    sigma0E*qRDotRho*(g*f*qRDotDRhoEK))/cFactorDr**2
            dRhoEK = self._dt*((dSigma0EEK+g*f*drEK)*self._a1-\
                              g*(1-f)*drEK*self._a2).dot(rhoAtN)+\
                     cFactorDRho.dot(dRhoEK)
            # End derivative wrt eK

            # Begin derivative wrt eX0
            qRDotDRhoEX0 = np.dot(self._qR, dRhoEX0)
            drEX0 = ((dSigma0EEX0*qRDotRho+sigma0E*qRDotDRhoEX0)*cFactorDr+\
                     sigma0E*qRDotRho*(g*f*qRDotDRhoEX0))/cFactorDr**2
            dRhoEX0 = self._dt*((dSigma0EEX0+g*f*drEX0)*self._a1-\
                              g*(1-f)*drEX0*self._a2).dot(rhoAtN)+\
                     cFactorDRho.dot(dRhoEX0)
            # End derivative wrt eEX0

            # Begin derivative wrt IL
            qRDotDRhoIL = np.dot(self._qR, dRhoIL)
            drIL = sigma0E*(qRDotDRhoIL*cFactorDr+qRDotRho*g*f*qRDotDRhoIL)/\
                   cFactorDr**2
            dRhoIL = self._dt*(g*f*drIL*self._a1-\
                              (dSigma0IIL+g*(1-f)*drIL)*self._a2).dot(rhoAtN)+\
                     cFactorDRho.dot(dRhoIL)
            # End derivative wrt IL

            # Begin derivative wrt IK
            qRDotDRhoIK = np.dot(self._qR, dRhoIK)
            drIK = sigma0E*(qRDotDRhoIK*cFactorDr+\
                               qRDotRho*g*f*qRDotDRhoIK)/\
                   cFactorDr**2
            dRhoIK = self._dt*(g*f*drIK*self._a1-\
                              (dSigma0IIK+g*(1-f)*drIK)*self._a2).dot(rhoAtN)+\
                     cFactorDRho.dot(dRhoIK)
            # End derivative wrt IK

            # Begin derivative wrt IX0
            qRDotDRhoIX0 = np.dot(self._qR, dRhoIX0)
            drIX0 = sigma0E*(qRDotDRhoIX0*cFactorDr+\
                               qRDotRho*g*f*qRDotDRhoIX0)/\
                   cFactorDr**2
            dRhoIX0 = self._dt*(g*f*drIX0*self._a1-\
                              (dSigma0IIX0+g*(1-f)*drIX0)*self._a2).dot(rhoAtN)+\
                     cFactorDRho.dot(dRhoIX0)
            # End derivative wrt IX0

            # Begin derivative wrt EF
            dSigma0EEF = eK*sigma0E*(1-sigma0E/eL)*sAtN
            qRDotDRhoEF = np.dot(self._qR, dRhoEF) 
            drEF = (dSigma0EEF*qRDotRho+sigma0E*qRDotDRhoEF*cFactorDr+\
                     sigma0E*qRDotRho*g*f*qRDotDRhoEF)/\
                   cFactorDr**2
            dRhoEF = self._dt*(np.outer(self._a1.dot(rhoAtN), 
                                         dSigma0EEF+g*f*drEF)-\
                                np.outer(self._a2.dot(rhoAtN), g*(1-f)*drEF))+\
                      cFactorDRho.dot(dRhoEF)
            # End derivative wrt EF

            # Begin derivative wrt IF
            dSigma0IIF = iK*sigma0I*(1-sigma0I/iL)*sAtN
            qRDotDRhoIF = np.dot(self._qR, dRhoIF) 
            drIF = (sigma0E*qRDotDRhoIF*cFactorDr+\
                     sigma0E*qRDotRho*g*f*qRDotDRhoIF)/\
                   cFactorDr**2
            dRhoIF = self._dt*(g*f*np.outer(self._a1.dot(rhoAtN), drIF)-\
                          np.outer(self._a2.dot(rhoAtN),
                                    dSigma0IIF+g*(1-f)*drIF))+\
                      cFactorDRho.dot(dRhoIF)
            # End derivative wrt IF

            drLeakages[n] = drLeakage
            drGs[n] = drG
            drFs[n] = drF
            drELs[n] = drEL
            drEKs[n] = drEK
            drEX0s[n] = drEX0
            drILs[n] = drIL
            drIKs[n] = drIK
            drIX0s[n] = drIX0
            drEFs[:,n] = drEF
            drIFs[:,n] = drIF

            dLLLeakage = dLLLeakage + (ys[n]-rs[n])*drLeakage
            dLLG = dLLG + (ys[n]-rs[n])*drG
            dLLF = dLLF + (ys[n]-rs[n])*drF
            dLLEL = dLLEL + (ys[n]-rs[n])*drEL
            dLLEK = dLLEK + (ys[n]-rs[n])*drEK
            dLLEX0 = dLLEX0 + (ys[n]-rs[n])*drEX0
            dLLIL = dLLIL + (ys[n]-rs[n])*drIL
            dLLIK = dLLIK + (ys[n]-rs[n])*drIK
            dLLIX0 = dLLIX0 + (ys[n]-rs[n])*drIX0
            dLLEF = dLLEF + (ys[n]-rs[n])*drEF
            dLLIF = dLLIF + (ys[n]-rs[n])*drIF
        return(dLLLeakage, dLLG, dLLF, dLLEL, dLLEK, dLLEX0, dLLIL, dLLIK, dLLIX0, dLLEF, dLLIF)

