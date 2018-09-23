
import sys
import math
import numpy as np
from scipy.optimize import fmin_cg
import pdb
from IFEDMWithInputLayerGradientCalculator import \
 IFEDMWithInputLayerGradientCalculator
from edmsMath import calculateLL

class IFEDMWithInputLayerMaximumLikelihoodOptimizer:

    def __init__(self, ifEDMWithInputLayer):
        self._ifEDMWithInputLayer = ifEDMWithInputLayer
        a0Tilde = ifEDMWithInputLayer.getA0Tilde()
        a1 = ifEDMWithInputLayer.getA1()
        a2 = ifEDMWithInputLayer.getA2()
        reversedQs = ifEDMWithInputLayer.getReversedQs()
        self._gradientCalculator = \
         IFEDMWithInputLayerGradientCalculator(a0Tilde=a0Tilde,
                                                a1=a1,
                                                a2=a2,
                                                reversedQs=reversedQs)
    def optimize(self, ys, inputs, t0, tf, dt, ysSigma, rho0,
                       optimizeLeakage=False, optimizeG=False, optimizeF=False,
                       optimizeEL=False, optimizeEK=False, optimizeEX0=False,
                       optimizeIL=False, optimizeIK=False, optimizeIX0=False,
                       optimizeEF=False, optimizeIF=False,
                       boundsLeakage=(1, 100), 
                       boundsG=(0, 100), 
                       boundsF=(0, 1),
                       boundsEL=(0, 10000), 
                       boundsEK=(0, 1), 
                       boundsEX0=(-1e6, 1e6),
                       boundsIL=(0, 10000), 
                       boundsIK=(0, 1), 
                       boundsIX0=(-1e6, 1e6),
                       boundsEF=((-1, -1, -1), (1, 1, 1)),
                       boundsIF=((-1, -1, -1), (1, 1, 1)),
                       stepSize=1e-3, tol=1e-4, maxIter=1000):

        def func(x):
            print("Evaluating func(x=%f)"%(x))
            leakage = x[0]
            self._ifEDMWithInputLayer.setLeakage(leakage=leakage)
            self._ifEDMWithInputLayer.prepareToIntegrate(t0=t0, tf=tf, dt=dt)
            self._ifEDMWithInputLayer.setInitialValue(rho0=rho0)
            times, rhos, rs, _ = self._ifEDMWithInputLayer.integrate()
            ll = calculateLL(ys=ys, rs=rs, ysSigma=ysSigma)
            return(-ll)

        def dFunc(x):
            print("Evaluating dFunc(x=%f)"%(x))
            leakage = x[0]
            self._ifEDMWithInputLayer.setLeakage(leakage=leakage)
            self._ifEDMWithInputLayer.prepareToIntegrate(t0=t0, tf=tf, dt=dt)
            self._ifEDMWithInputLayer.setInitialValue(rho0=rho0)
            times, rhos, rs, _ = self._ifEDMWithInputLayer.integrate()
            g = self._ifEDMWithInputLayer.getG()
            f = self._ifEDMWithInputLayer.getF()
            eL = self._ifEDMWithInputLayer.getEL()
            eK = self._ifEDMWithInputLayer.getEK()
            eX0 = self._ifEDMWithInputLayer.getEX0()
            iL = self._ifEDMWithInputLayer.getEL()
            iK = self._ifEDMWithInputLayer.getIK()
            iX0 = self._ifEDMWithInputLayer.getIX0()
            eF = self._ifEDMWithInputLayer.getEFilter()
            iF = self._ifEDMWithInputLayer.getIFilter()
            dLLLeakages, dLLGs, dLLFs, \
             dLLELs, dLLEKs, dLLEX0s, \
             dLLILs, dLLIKs, dLLIX0s, \
             dLLEFs, dLLIFs = self._gradientCalculator.\
                               deriv(ys=ys, rs=rs, rhos=rhos, inputs=inputs,
                                            dt=dt, ysSigma=ysSigma, 
                                            leakage=leakage, g=g, f=f, 
                                            eL=eL, eK=eK, eX0=eX0, 
                                            iL=iL, iK=iK, iX0=iX0, 
                                            eF=eF, iF=iF,
                                            computeDLeakage=optimizeLeakage,
                                            computeDG=optimizeG,
                                            computeDF=optimizeF,
                                            computeDEL=optimizeEL,
                                            computeDEK=optimizeEK,
                                            computeDEX0=optimizeEX0,
                                            computeDIL=optimizeIL,
                                            computeDIK=optimizeIK,
                                            computeDIX0=optimizeIX0,
                                            computeDEF=optimizeEF,
                                            computeDIF=optimizeIF)
            return(-dLLLeakages[-1])

        minValue, fAtMinValue = \
         fmin_cg(f=func, 
                  x0=np.asarray(self._ifEDMWithInputLayer.getLeakage()),
                  fprime=dFunc)
        pdb.set_trace()

