
import sys
import pdb
import numpy as np

class IFEDMsSimpleGradientCalculatorRWFWI:

    def __init__(self, a0Tilde, a1, a2, dt, dv, reversedQs):
        self._a0Tilde = a0Tilde
        self._a1 = a1
        self._a2 = a2
        self._dt = dt
        self._dv = dv
        self._reversedQs = reversedQs

    def deriv(self, ys, rs, rhos, eInputs, iInputs, ysSigma, leakage, g, f):
        dRhoLeakage = np.zeros(self._a0Tilde.shape[0])
        drLeakage = 0
        idM = np.identity(self._a0Tilde.shape[0])
        dLLLeakage = 0
        dLLLeakages = np.zeros(eInputs.size)
        dLLLeakages[0] = dLLLeakage

        for n in xrange(1, eInputs.size):
            if n%1==0:
                print("Gradient calculation step %d (out of %d)" % 
                      (n, eInputs.size))
                print("dLLLeakage=%f, drLeakage=%f"%(dLLLeakage, drLeakage))
                sys.stdout.flush()
            rhoAtN = rhos[:,n]
            rAtN = rs[n]
            cFactorDRho = idM+self._dt*(leakage/2*self._a0Tilde+
                                         (eInputs[n]+g*f*rAtN)*self._a1-
                                         (iInputs[n]+g*(1-f)*rAtN)*self._a2)
            qRDotRho = self._dv*self._reversedQs.dot(rhoAtN)
            cFactorDr = 1-g*f*qRDotRho

            # begin deriv leakage
            qRDotDRhoLeakage = self._dv*self._reversedQs.dot(dRhoLeakage)
            drLeakage = eInputs[n]*(qRDotDRhoLeakage*cFactorDr+\
                                     qRDotRho*g*f*qRDotDRhoLeakage)/cFactorDr**2
            dRhoLeakage = self._dt*(0.5*self._a0Tilde+g*f*drLeakage*self._a1-\
                                    g*(1-f)*drLeakage*self._a2).dot(rhoAtN)+\
                            cFactorDRho.dot(dRhoLeakage)
            dLLLeakage = dLLLeakage + (ys[n]-rs[n])*drLeakage
            dLLLeakages[n] = dLLLeakage
            # end deriv leakage

        dLLLeakages = dLLLeakages/ysSigma**2
        return(dLLLeakages)

