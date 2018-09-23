
import sys
import math
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
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
                       boundsLeakage=(1.0, 100.0), 
                       boundsG=(0.0, 100.0), 
                       boundsF=(0.0, 1.0),
                       boundsEL=(0.0, 10000.0), 
                       boundsEK=(0.0, 1.0), 
                       boundsEX0=(-1e6, 1e6),
                       boundsIL=(0.0, 10000.0), 
                       boundsIK=(0.0, 1.0), 
                       boundsIX0=(-1e6, 1e6),
                       boundsEF=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                       boundsIF=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                       stepSize=1e-3, tol=1e-4, maxIter=1000):

        def func(x):
            print("Evaluating func(x=%f)"%(x))
            leakage = x[0]
            self._ifEDMWithInputLayer.setLeakage(leakage=leakage)
            self._ifEDMWithInputLayer.prepareToIntegrate(t0=t0, tf=tf, dt=dt)
            self._ifEDMWithInputLayer.setInitialValue(rho0=rho0)
            times, rhos, rs, _ = self._ifEDMWithInputLayer.integrate()
            ll = calculateLL(ys=ys, rs=rs, ysSigma=ysSigma)
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
            return(-ll, np.array([-dLLLeakages[-1]]))

        pdb.set_trace()
        minValue, fAtMinValue, info = \
         fmin_l_bfgs_b(func=func, 
                  x0=np.array([self._ifEDMWithInputLayer.getLeakage()]),
                  bounds=[boundsLeakage],
                  iprint=1)
        pdb.set_trace()

