
import sys
import pdb
import math
import numpy as np
from scipy.optimize.tnc import MSG_ALL, RCSTRINGS, fmin_tnc
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
from edmsMath import calculateELLs
from ifEDMsFunctions import buildRho0
from IFEDMWithSinusoidalInputGradientsCalculator import \
 IFEDMWithSinusoidalInputGradientsCalculator
from BetaRho0Calculator import BetaRho0Calculator

class IFEDMWithSinusoidalInputMaximumLikelihoodOptimizer:

    def __init__(self, ifEDMWithSinusoidalInput, rho0A, rho0B,
                       variablesToOptimize, bounds):
        self._ifEDMWithSinusoidalInput = ifEDMWithSinusoidalInput
        self._rho0A = rho0A
        self._rho0B = rho0B
        self._variablesToOptimize = variablesToOptimize
        self._bounds = bounds
        a0Tilde = ifEDMWithSinusoidalInput.getA0Tilde()
        a1 = ifEDMWithSinusoidalInput.getA1()
        a2 = ifEDMWithSinusoidalInput.getA2()
        reversedQs = ifEDMWithSinusoidalInput.getReversedQs()
        fixedParameterValues = \
         self._getFixedParameterValues(ifEDMWithSinusoidalInput=\
                                         self._ifEDMWithSinusoidalInput,
                                        rho0A=rho0A,
                                        rho0B=rho0B,
                                        variablesToOptimize=
                                         variablesToOptimize)
        self._gradientsCalculator = \
         IFEDMWithSinusoidalInputGradientsCalculator(a0Tilde=a0Tilde,
                                                      a1=a1,
                                                      a2=a2,
                                                      reversedQs=reversedQs,
                                                      variablesToOptimize=
                                                       variablesToOptimize,
                                                      fixedParameterValues=
                                                       fixedParameterValues)

    def _saveCurrentParameterVector(self, xk):
        self._parameterVectors.append(xk)

    def optimize_bfgs(self, ys, ysSigma, t0, tf, dt,
                            scale=1.0, timeToOptimize=None, 
                            thrZeroNorm=np.finfo(np.float).eps, 
                            nStepsBtwFuncPrintouts=1000,
                            nStepsBtwGradientPrintouts=1000,
                            gtol=-1, norm=np.inf,
                            maxiter=None,
                            full_output=1, disp=1, retall=1, 
                            callback=None):
        if timeToOptimize is None:
            timeToOptimize = tf-dt

        if callback is None:
            callback = self._saveCurrentParameterVector

        self._ys = ys
        self._ysSigma = ysSigma
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._timeToOptimizeIndex = round((timeToOptimize-t0)/dt)
        self._scale = scale
        self._thrZeroNorm = thrZeroNorm
        self._nStepsBtwFuncPrintouts = nStepsBtwFuncPrintouts
        self._nStepsBtwGradientPrintouts = nStepsBtwGradientPrintouts
        self._lastIntegratedX = None
        self._parameterVectors = []

        x0, bounds = self._getX0AndBounds(ifEDMWithSinusoidalInput=\
                                            self._ifEDMWithSinusoidalInput,
                                           rho0A=self._rho0A,
                                           rho0B=self._rho0B,
                                           variablesToOptimize=
                                            self._variablesToOptimize,
                                           bounds=self._bounds)
        if full_output and retall:
            xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag, \
             allvecs = fmin_bfgs(f=self._optimizationFunction, 
                                  x0=np.array(x0),
                                  fprime=self._gradientFunction,
                                  gtol=gtol,
                                  norm=norm,
                                  callback=callback,
                                  maxiter=maxiter,
                                  full_output=full_output,
                                  disp=disp,
                                  retall=retall)
            return(xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag,
                         allvecs, self._parameterVectors)
        elif full_output and not retall:
            xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = \
             fmin_bfgs(f=self._optimizationFunction, 
                       x0=np.array(x0),
                       fprime=self._gradientFunction,
                       gtol=gtol,
                       norm=norm,
                       callback=callback,
                       maxiter=maxiter,
                       full_output=full_output,
                       disp=disp,
                       retall=retall)
            return(xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag,
                         self._parameterVectors)
        else:
            xopt = fmin_bfgs(f=self._optimizationFunction, 
                              x0=np.array(x0),
                              fprime=self._gradientFunction,
                              gtol=gtol,
                              norm=norm,
                              callback=callback,
                              maxiter=maxiter,
                              full_output=full_output,
                              disp=disp,
                              retall=retall)
            return(xopt, self._parameterVectors)

    def optimize_fmin_tc(self, ys, ysSigma, t0, tf, dt,
                               scale=1.0, timeToOptimize=None,
                               thrZeroNorm=np.finfo(np.float).eps, 
                               nStepsBtwFuncPrintouts=1000,
                               nStepsBtwGradientPrintouts=1000,
                               scale_fmin_tc=None, offset=None, 
                               messages=MSG_ALL, disp=5, maxCGit=-1, 
                               maxfun=None, eta=-1, stepmx=0, accuracy=0, 
                               fmin=0, ftol=-1, xtol=1e-6, pgtol=-1, rescale=0):
#                        callback=_saveCurrentParameterVector):
        if timeToOptimize is None:
            timeToOptimize = tf-dt

        self._ys = ys
        self._ysSigma = ysSigma
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._timeToOptimizeIndex = round((timeToOptimize-t0)/dt)
        self._scale = scale
        self._thrZeroNorm = thrZeroNorm
        self._nStepsBtwFuncPrintouts = nStepsBtwFuncPrintouts
        self._nStepsBtwGradientPrintouts = nStepsBtwGradientPrintouts
        self._lastIntegratedX = None
        self._parameterVectors = []

        x0, bounds = self._getX0AndBounds(ifEDMWithSinusoidalInput=
                                            self._ifEDMWithSinusoidalInput,
                                           rho0A=self._rho0A,
                                           rho0B=self._rho0B,
                                           variablesToOptimize=
                                            self._variablesToOptimize,
                                           bounds=self._bounds)
        x, nfeval, rc = \
         fmin_tnc(func=self._optimizationFunction, 
                   x0=np.array(x0),
                   fprime=self._gradientFunction,
                   bounds=bounds,
                   scale=scale_fmin_tc,
                   offset=offset,
                   messages=messages,
                   maxCGit=maxCGit,
                   maxfun=maxfun,
                   eta=eta,
                   stepmx=stepmx,
                   accuracy=accuracy,
                   fmin=fmin,
                   ftol=ftol,
                   xtol=xtol,
                   pgtol=pgtol,
                   rescale=rescale)
#                    callback=callback)
        return(x, nfeval, RCSTRINGS[rc], self._parameterVectors)

    def _optimizationFunction(self, x):
        print("Evaluation at %s"%(str(x)))
        if self._lastIntegratedX is None or \
           np.linalg.norm(x-self._lastIntegratedX)>self._thrZeroNorm:
            self._integrateEDM(x=x)
        eLLs = calculateELLs(ys=self._ys, rs=self._rs, ysSigma=self._ysSigma)
        answer = -eLLs[self._timeToOptimizeIndex]*self._scale
        print("Result evaluation at %s=%f"%(str(x), answer))
        return(answer)

    def _gradientFunction(self, x):
        print("Gradient at %s"%(str(x)))
        if self._lastIntegratedX is None or \
           np.linalg.norm(x-self._lastIntegratedX)>self._thrZeroNorm:
            self._integrateEDM(x=x)
        deriv = self._gradientsCalculator.deriv(x=x, ys=self._ys, 
                                                     rs=self._rs,
                                                     rhos=self._rhos,
                                                     ysSigma=self._ysSigma,
                                                     times=self._times,
                                                     nStepsBtwPrintouts=
                                                      self._nStepsBtwGradientPrintouts)
        derivAtTimeToOptimize = np.empty(len(deriv))
        for i in xrange(len(deriv)):
            derivAtTimeToOptimize[i] = deriv[i][self._timeToOptimizeIndex]
        answer = -derivAtTimeToOptimize*self._scale
        print("Result gradient at %s=%s"%(str(x), str(answer)))
        return(answer)


    def _integrateEDM(self, x):
        self._setEDMParameterValues(values=x, variablesToOptimize=\
                                               self._variablesToOptimize)
        self._ifEDMWithSinusoidalInput.prepareToIntegrate(t0=self._t0,
                                                           tf=self._tf,
                                                           dt=self._dt)
        nVSteps = self._ifEDMWithSinusoidalInput.getA0Tilde().shape[0]
        rho0 = BetaRho0Calculator(nVSteps=nVSteps).getRho0(a=self._rho0A,
                                                            b=self._rho0B)
        self._ifEDMWithSinusoidalInput.setInitialValue(rho0=rho0)
        self._times, self._rhos, self._rs, _, eInputs, _, iInputs, _ = \
         self._ifEDMWithSinusoidalInput.integrate(nStepsBtwPrintouts=self._nStepsBtwFuncPrintouts)

        # Begin delete
        # eInput = self._ifEDMWithSinusoidalInput._eSinusoidal.eval(t=self._times)
        # iInput = self._ifEDMWithSinusoidalInput._iSinusoidal.eval(t=self._times)
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(self._times, self._ys, color="blue", label="Data")
        plt.plot(self._times, self._rs, color="red", linewidth=2.0,
                              label="Estimate")
        plt.xlabel("Time (sec)")
        plt.ylabel("Scaled High-Gamma Power")
        plt.title(x, fontsize="small")
        plt.legend(loc="upper left")

        self._ax = plt.twinx()
        self._ax.plot(self._times, eInputs, color="green", linestyle="--",
                                   label="eInput")
        self._ax.plot(self._times, iInputs, color="magenta", linestyle="--",
                                   label="iInput")
        self._ax.set_ylabel("Input Spike Rate (ips)")
        self._ax.legend(loc="upper right")

        betaRho0Calculator = BetaRho0Calculator(nVSteps=nVSteps)
        rho0 = betaRho0Calculator.getRho0(a=self._rho0A, b=self._rho0B)
        vs = betaRho0Calculator.getVs()
        plt.subplot(2, 1, 2)
        plt.plot(vs, rho0)
        plt.xlabel("Normalized Votage")
        plt.ylabel("Probability Density")

        plt.draw()
        # End delete

        self._lastIntegratedX = np.copy(a=x)

    def _setEDMParameterValues(self, values, variablesToOptimize):
        valuesIndex = 0

        if "leakage" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setLeakage(leakage=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "g" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setG(g=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "f" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setF(f=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "rho0A" in variablesToOptimize:
            self._rho0A = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "rho0B" in variablesToOptimize:
            self._rho0B = values[valuesIndex]
            valuesIndex =  valuesIndex + 1

        if "eDC" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setEDC(eDC=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "eAmpl" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setEAmpl(eAmpl=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "eFreq" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setEFreq(eFreq=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "ePhase" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setEPhase(ePhase=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "iDC" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setIDC(iDC=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "iAmpl" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setIAmpl(iAmpl=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "iFreq" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setIFreq(iFreq=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

        if "iPhase" in variablesToOptimize:
            self._ifEDMWithSinusoidalInput.setIPhase(iPhase=values[valuesIndex])
            valuesIndex =  valuesIndex + 1

    def _getX0AndBounds(self, ifEDMWithSinusoidalInput, rho0A, rho0B, variablesToOptimize, bounds):
        x0 = []
        answerBounds = []

        if "leakage" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getLeakage())
            answerBounds.append(bounds["leakage"])

        if "g" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getG())
            answerBounds.append(bounds["g"])

        if "f" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getF())
            answerBounds.append(bounds["f"])

        if "rho0A" in variablesToOptimize:
            x0.append(rho0A)
            answerBounds.append(bounds["rho0A"])

        if "rho0B" in variablesToOptimize:
            x0.append(rho0B)
            answerBounds.append(bounds["rho0B"])

        if "eDC" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getEDC())
            answerBounds.append(bounds["eDC"])

        if "eAmpl" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getEAmpl())
            answerBounds.append(bounds["eAmpl"])

        if "eFreq" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getEFreq())
            answerBounds.append(bounds["eFreq"])

        if "ePhase" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getEPhase())
            answerBounds.append(bounds["ePhase"])

        if "iDC" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getIDC())
            answerBounds.append(bounds["iDC"])

        if "iAmpl" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getIAmpl())
            answerBounds.append(bounds["iAmpl"])

        if "iFreq" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getIFreq())
            answerBounds.append(bounds["iFreq"])

        if "iPhase" in variablesToOptimize:
            x0.append(ifEDMWithSinusoidalInput.getIPhase())
            answerBounds.append(bounds["iPhase"])

        return(x0, answerBounds)

    def _getFixedParameterValues(self, ifEDMWithSinusoidalInput, 
                                       rho0A, rho0B,
                                       variablesToOptimize):
        fixedParameterValues = []

        if "leakage" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getLeakage())

        if "g" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getG())

        if "f" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getF())

        if "rho0A" not in variablesToOptimize:
            fixedParameterValues.append(rho0A)

        if "rho0B" not in variablesToOptimize:
            fixedParameterValues.append(rho0B)

        if "eDC" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getEDC())

        if "eAmpl" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getEAmpl())

        if "eFreq" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getEFreq())

        if "ePhase" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getEPhase())

        if "iDC" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getIDC())

        if "iAmpl" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getIAmpl())

        if "iFreq" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getIFreq())

        if "iPhase" not in variablesToOptimize:
            fixedParameterValues.append(ifEDMWithSinusoidalInput.getIPhase())

        return(fixedParameterValues)

