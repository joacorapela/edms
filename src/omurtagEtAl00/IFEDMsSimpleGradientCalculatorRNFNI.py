
import sys
import pdb
import numpy as np

class IFEDMsSimpleGradientCalculator:

    def __init__(self, a0Tilde, a1, dt, dv, reversedQs):
        self._a0Tilde = a0Tilde
        self._a1 = a1
        self._dt = dt
        self._dv = dv
        self._reversedQs = reversedQs

    def deriv(self, ys, rs, rhos, inputs, ysSigma, leakage):
        dRhoLeakage = np.zeros(self._a0Tilde.shape[0])
        drLeakage = 0
        idM = np.identity(self._a0Tilde.shape[0])
        dLLLeakage = 0
        dLLLeakages = np.zeros(inputs.size)
        dLLLeakages[0] = dLLLeakage

        for n in xrange(1, inputs.size):
            if n%1==0:
                print("Gradient calculation step %d (out of %d)" % 
                      (n, inputs.size))
                print("dLLLeakage=%f, drLeakage=%f"%(dLLLeakage, drLeakage))
                sys.stdout.flush()
            rhoAtN = rhos[:,n]
            cFactorDRho = idM+self._dt*(leakage/2*self._a0Tilde+
                                         inputs[n]*self._a1)
            drLeakage = inputs[n]*self._dv*self._reversedQs.dot(dRhoLeakage)
            dRhoLeakage = self._dt/2*self._a0Tilde.dot(rhoAtN)+\
                           cFactorDRho.dot(dRhoLeakage)
            dLLLeakage = dLLLeakage + (ys[n]-rs[n])*drLeakage
            dLLLeakages[n] = dLLLeakage

        dLLLeakages = dLLLeakages/ysSigma**2
        return(dLLLeakages)

