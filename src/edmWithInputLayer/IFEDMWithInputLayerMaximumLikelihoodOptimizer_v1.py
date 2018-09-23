
import sys
import math
import numpy as np
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
    def _getNewVariableValues(self, variable, boundsVariable, dLLVariable, 
                                    normGradient, stepSize):
        tmpVariable = variable + stepSize * dLLVariable
        if tmpVariable<boundsVariable[0]:
            dLLVariable = (boundsVariable[0]-variable)/stepSize
            variable = boundsVariable[0]
            stepSize = stepSize/2
        elif boundsVariable[1]<tmpVariable:
            dLLVariable = (boundsVariable[1]-variable)/stepSize
            variable = boundsVariable[1]
            stepSize = stepSize/2
        else:
            variable = tmpVariable
        normGradient = normGradient + dLLVariable**2
        return(variable, normGradient, stepSize)

    def _getNewFilterVariablesValues(self, filter, boundsFilter, dLLFilter, 
                                           normGradient, stepSize):
        tmpFilter = filter + stepSize * dLLFilter
        stepSizeUpdated = False
        for j in xrange(len(filter)):
            if tmpFilter[j]<boundsFilter[0][j]:
                dLLFilter[j] = (boundsFilter[0][j]-filter[j])/stepSize
                filter[j] = boundsFilter[0][j]
                if not stepSizeUpdated:
                    stepSize = stepSize/2
            elif boundsFilter[1][j]<tmpFilter[j]:
                dLLFilter[j] = (boundsFilter[1][j]-filter[j])/stepSize
                filter[j] = boundsFilter[1][j]
                if not stepSizeUpdated:
                    stepSize = stepSize/2
            else:
                filter[j] = tmpFilter[j]
        normGradient = normGradient + (dLLFilter**2).sum()
        return(filter, normGradient, stepSize)

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

        def f(leakage):
            self._ifEDMWithInputLayer.setLeakage(leakage=leakage)
            self._ifEDMWithInputLayer.prepareToIntegrate(t0=t0, tf=tf, dt=dt)
            self._ifEDMWithInputLayer.setInitialValue(rho0=rho0)
            times, rhos, rs, _ = self._ifEDMWithInputLayer.integrate()
            ll = calculateLL(ys=ys, rs=rs, ysSigma=ysSigma)
            return(ll)

        def fprime(leakage):
            self._ifEDMWithInputLayer.setLeakage(leakage=leakage)
            self._ifEDMWithInputLayer.prepareToIntegrate(t0=t0, tf=tf, dt=dt)
            self._ifEDMWithInputLayer.setInitialValue(rho0=rho0)
            times, rhos, rs, _ = self._ifEDMWithInputLayer.integrate()
            ll = calculateLL(ys=ys, rs=rs, ysSigma=ysSigma)

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
            return(dLLLeakages)

        # y?s should be sampled at dtSaveRhos
        leakage = self._ifEDMWithInputLayer.getLeakage()
        g   = self._ifEDMWithInputLayer.getG()
        f   = self._ifEDMWithInputLayer.getF()
        eL  = self._ifEDMWithInputLayer.getEL()
        eK  = self._ifEDMWithInputLayer.getEK()
        eX0 = self._ifEDMWithInputLayer.getEX0()
        iL  = self._ifEDMWithInputLayer.getIL()
        iK  = self._ifEDMWithInputLayer.getIK()
        iX0 = self._ifEDMWithInputLayer.getIX0()
        eF  = self._ifEDMWithInputLayer.getEFilter()
        iF  = self._ifEDMWithInputLayer.getIFilter()
        normGradient = float("Inf")
        iterNo = 0
        lls = np.empty(maxIter)
        lls[iterNo] = -float("Inf")

        while normGradient>tol and iterNo<maxIter:
            iterNo = iterNo + 1
            print("Iteration %d, leakage=%f, g=%f, f=%f, eL=%f, eK=%f, eX0=%f, iL=%f, iK=%f, iX0=%f, eF=%s, iF=%s, |Grad|=%f, ll=%.02f" % \
                  (iterNo, leakage, g, f, eL, eK, eX0, iL, iK, iX0, str(eF),
                           str(iF), normGradient, lls[iterNo]))
            sys.stdout.flush()

            self._ifEDMWithInputLayer.prepareToIntegrate(t0=t0, tf=tf, dt=dt)
            self._ifEDMWithInputLayer.setInitialValue(rho0=rho0)
            times, rhos, rs, _ = self._ifEDMWithInputLayer.integrate()

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
            lls[iterNo] = calculateLL(ys=ys, rs=rs, ysSigma=ysSigma)

            normGradient = 0.0

            if optimizeLeakage:
                pdb.set_trace()
                leakage, normGradient, stepSize = \
                 self._getNewVariableValues(variable=leakage,
                                             boundsVariable=boundsLeakage,
                                             dLLVariable=dLLLeakages[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setLeakage(leakage=leakage)
            if optimizeG:
                g, normGradient, stepSize = \
                 self._getNewVariableValues(variable=g,
                                             boundsVariable=boundsG,
                                             dLLVariable=dLLGs[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setG(g=g)
            if optimizeF:
                f, normGradient, stepSize = \
                 self._getNewVariableValues(variable=f,
                                             boundsVariable=boundsF,
                                             dLLVariable=dLLFs[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setF(f=f)
            if optimizeEL:
                eL, normGradient, stepSize = \
                 self._getNewVariableValues(variable=eL,
                                             boundsVariable=boundsEL,
                                             dLLVariable=dLLELs[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setEL(eL=eL)
            if optimizeEK:
                eK, normGradient, stepSize = \
                 self._getNewVariableValues(variable=eK,
                                             boundsVariable=boundsEK,
                                             dLLVariable=dLLEKs[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setEK(eK=eK)
            if optimizeEX0:
                eX0, normGradient, stepSize = \
                 self._getNewVariableValues(variable=eX0,
                                             boundsVariable=boundsEX0,
                                             dLLVariable=dLLEX0s[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setEX0(eX0=eX0)
            if optimizeIL:
                il, normGradient, stepSize = \
                 self._getNewVariableValues(variable=il,
                                             boundsVariable=boundsIL,
                                             dLLVariable=dLLILs[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setIL(iL=iL)
            if optimizeIK:
                iK, normGradient, stepSize = \
                 self._getNewVariableValues(variable=iK,
                                             boundsVariable=boundsIK,
                                             dLLVariable=dLLIKs[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setIK(iK=iK)
            if optimizeIX0:
                iX0, normGradient, stepSize = \
                 self._getNewVariableValues(variable=iX0,
                                             boundsVariable=boundsIX0,
                                             dLLVariable=dLLIX0s[-1],
                                             normGradient=normGradient,
                                             stepSize=stepSize)
                self._ifEDMWithInputLayer.setIX0(iX0=iX0)
            if optimizeEF:
                eF, normGradient, stepSize = \
                 self._getNewFilterVariablesValues(filter=eF,
                                                    boundsFilter=boundsEF,
                                                    dLLFilter=dLLEFs[:,-1],
                                                    normGradient=normGradient,
                                                    stepSize=stepSize)
                self._ifEDMWithInputLayer.setEFilter(filter=eF)
            if optimizeIF:
                iF, normGradient, stepSize = \
                 self._getNewFilterVariablesValues(filter=iF,
                                                    boundsFilter=boundsIF,
                                                    dLLFilter=dLLIFs[:,-1],
                                                    normGradient=normGradient,
                                                    stepSize=stepSize)
                self._ifEDMWithInputLayer.setIFilter(filter=iF)

        if normGradient<tol:
            converged = True
        else:
            converged = False
        return(leakage, g, f, eL, eK, eX0, iL, iK, iX0, eF, iF, converged, lls)

