
import sys
import math
import numpy as np
from scipy.optimize import fmin_cg
import pdb
from IFEDMWithInputLayerGradientCalculator import \
 IFEDMWithInputLayerGradientCalculator
from edmsMath import calculateELLs

class IFEDMWithInputLayerMaximumLikelihoodOptimizer:

    def __init__(self, ifEDMWithInputLayer, variablesToOptimize, bounds):
        self._ifEDMWithInputLayer = ifEDMWithInputLayer
        self._variablesToOptimize = variablesToOptimize
        self._bounds = bounds
        a0Tilde = ifEDMWithInputLayer.getA0Tilde()
        a1 = ifEDMWithInputLayer.getA1()
        a2 = ifEDMWithInputLayer.getA2()
        reversedQs = ifEDMWithInputLayer.getReversedQs()
        fixedParameterValues = \
         self._getFixedParameterValues(ifEDMWithInputLayer=
                                         self._ifEDMWithInputLayer,
                                        variablesToOptimize=variablesToOptimize)
        eFilterSize = ifEDMWithInputLayer.getEFilter().size
        iFilterSize = ifEDMWithInputLayer.getIFilter().size
        self._gradientCalculator = \
         IFEDMWithInputLayerGradientCalculator(a0Tilde=a0Tilde,
                                                a1=a1,
                                                a2=a2,
                                                reversedQs=reversedQs,
                                                eFilterSize=eFilterSize,
                                                iFilterSize=iFilterSize,
                                                variablesToOptimize=
                                                 variablesToOptimize,
                                                fixedParameterValues=
                                                 fixedParameterValues)

    def optimize(self, ys, inputs, t0, tf, dt, ysSigma, rho0, timeToOptimize,
                       scale=10000, thrZeroNorm=np.finfo(np.float).eps, 
                       gtol=1e-05, norm=float("inf"), maxiter=None, 
                       full_output=True, disp=True, retall=True):
        self._ys = ys
        self._inputs = inputs
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._ysSigma = ysSigma
        self._rho0 = rho0
        self._timeToOptimizeIndex = round((timeToOptimize-t0)/dt)
        self._scale = scale
        self._thrZeroNorm = thrZeroNorm
        self._lastIntegratedX = None

        x0, bounds = self._getX0AndBounds(ifEDMWithInputLayer=
                                            self._ifEDMWithInputLayer,
                                           variablesToOptimize=
                                            self._variablesToOptimize)
        xopt, fopt, func_calls, grad_calls, warnflag, allvecs = \
         fmin_cg(f=self._optimizationFunction, 
                  x0=np.array(x0),
                  gtol=gtol,
                  fprime=self._gradientFunction,
                  norm=norm,
                  maxiter=maxiter,
                  full_output=full_output,
                  disp=disp,
                  retall=retall)
        return(xopt, fopt, allvecs)

    def _optimizationFunction(self, x):
        print("Evaluation at %s"%(str(x)))
        if self._lastIntegratedX is None or \
           np.linalg.norm(x-self._lastIntegratedX)>self._thrZeroNorm:
            self._integrateEDM(x=x)
        eLLs = calculateELLs(ys=self._ys, rs=self._rs, ysSigma=self._ysSigma)
        answer = -eLLs[self._timeToOptimizeIndex]/self._scale
        print("Evaluation at %s=%f"%(str(x), answer))
        return(answer)

    def _gradientFunction(self, x):
        print("Gradient at %s"%(str(x)))
        if self._lastIntegratedX is None or \
           np.linalg.norm(x-self._lastIntegratedX)>self._thrZeroNorm:
            self._integrateEDM(x=x)
        deriv = self._gradientCalculator.deriv(x=x, ys=self._ys, 
                                                  rs=self._rs,
                                                  rhos=self._rhos,
                                                  inputs=self._inputs,
                                                  ysSigma=self._ysSigma,
                                                  dt=self._dt)
        derivAtTimeToOptimize = np.empty(len(deriv))
        for i in xrange(len(deriv)):
            derivAtTimeToOptimize[i] = deriv[i][self._timeToOptimizeIndex]
        answer = -derivAtTimeToOptimize/self._scale
        print("Gradient at %s=%s"%(str(x), str(answer)))
        return(answer)

    def _integrateEDM(self, x):
        self._setEDMParameterValues(ifEDMWithInputLayer=
                                      self._ifEDMWithInputLayer,
                                     values=x,
                                     variablesToOptimize=
                                      self._variablesToOptimize)
        self._ifEDMWithInputLayer.prepareToIntegrate(t0=self._t0, 
                                                      tf=self._tf,
                                                      dt=self._dt)
        self._ifEDMWithInputLayer.setInitialValue(rho0=self._rho0)
        self._times, self._rhos, self._rs, _ = \
         self._ifEDMWithInputLayer.integrate()
        self._lastIntegratedX = x

    def _setEDMParameterValues(self, ifEDMWithInputLayer, values, 
                                     variablesToOptimize):
        valuesIndex = 0

        if "leakage" in variablesToOptimize:
            ifEDMWithInputLayer.setLeakage(leakage=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "g" in variablesToOptimize:
            ifEDMWithInputLayer.setG(g=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "f" in variablesToOptimize:
            ifEDMWithInputLayer.setF(f=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "eL" in variablesToOptimize:
            ifEDMWithInputLayer.setEL(eL=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "eK" in variablesToOptimize:
            ifEDMWithInputLayer.setEK(eK=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "eX0" in variablesToOptimize:
            ifEDMWithInputLayer.setEX0(eX0=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "iL" in variablesToOptimize:
            ifEDMWithInputLayer.setIL(il=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "iK" in variablesToOptimize:
            ifEDMWithInputLayer.setIK(iK=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "iX0" in variablesToOptimize:
            ifEDMWithInputLayer.setIX0(iX0=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "eF" in variablesToOptimize:
            iFilterSize = ifEDMWithInputLayer.getIFilter().size
            iFilter = np.array(values[valuesIndex:(valuesIndex+iFilterSize)])
            ifEDMWithInputLayer.setIFilter(filter=iFilter)
            valuesIndex =  valuesIndex + iFilterSize

        if "iF" in variablesToOptimize:
            eFilterSize = ifEDMWithInputLayer.getIFilter().size
            eFilter = np.array(values[valuesIndex:(valuesIndex+eFilterSize)])
            ifEDMWithInputLayer.setIFilter(filter=eFilter)
            valuesIndex =  valuesIndex + eFilterSize

    def _getX0AndBounds(self, ifEDMWithInputLayer, variablesToOptimize):
        x0 = []
        answerBounds = []

        if "leakage" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getLeakage())
            answerBounds.append(self._bounds["leakage"])

        if "g" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getG())
            answerBounds.append(self._bounds["g"])

        if "f" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getF())
            answerBounds.append(self._bounds["f"])

        if "eL" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getEL())
            answerBounds.append(self._bounds["eL"])

        if "eK" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getEK())
            answerBounds.append(self._bounds["eK"])

        if "eX0" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getEX0())
            answerBounds.append(self._bounds["eX0"])

        if "iL" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getIL())
            answerBounds.append(self._bounds["iL"])

        if "iK" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getIK())
            answerBounds.append(self._bounds["iK"])

        if "iX0" in variablesToOptimize:
            x0.append(ifEDMWithInputLayer.getIX0())
            answerBounds.append(self._bounds["iX0"])

        if "eF" in variablesToOptimize:
            x0.extend(ifEDMWithInputLayer.getEFilter().tolist())
            answerBounds.extend(self._bounds["eF"])

        if "iF" in variablesToOptimize:
            x0.extend(ifEDMWithInputLayer.getIFilter().tolist())
            answerBounds.extend(self._bounds["iF"])

        return(x0, answerBounds)

    def _getFixedParameterValues(self, ifEDMWithInputLayer, 
                                       variablesToOptimize):
        fixedParameterValues = []

        if "leakage" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getLeakage())

        if "g" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getG())

        if "f" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getF())

        if "eL" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getEL())

        if "eK" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getEK())

        if "eX0" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getEX0())

        if "iL" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getIL())

        if "iK" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getIK())

        if "iX0" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithInputLayer.getIX0())

        if "eF" not in variablesToOptimize:
            fixedParameterValues.extend(ifEDMWithInputLayer.getEFilter().\
                                         tolist())

        if "iF" not in variablesToOptimize:
            fixedParameterValues.extend(ifEDMWithInputLayer.getIFilter().\
                                         tolist())

        return(fixedParameterValues)

