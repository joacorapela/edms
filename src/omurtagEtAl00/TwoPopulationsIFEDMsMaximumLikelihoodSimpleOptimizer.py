
import sys
import math
import numpy as np
import pdb
from TwoPopulationsIFEnsembleDensitySimpleIntegrator import TwoPopulationsIFEnsembleDensitySimpleIntegrator
from TwoPopulationsIFEDMsSimpleGradientCalculator import TwoPopulationsIFEDMsSimpleGradientCalculator
from TwoPopulationsLLCalculator import TwoPopulationsLLCalculator

class TwoPopulationsIFEDMsMaximumLikelihoodSimpleOptimizer:

    def __init__(self, a0, a1, a2, dt, dv, reversedQs):
        self._a0 = a0
        self._a1 = a1
        self._a2 = a2
        self._twoPIntegrator = \
         TwoPopulationsIFEnsembleDensitySimpleIntegrator(a0=a0, a1=a1, a2=a2, 
                                                            dt=dt, dv=dv, 
                                                            reversedQs=
                                                             reversedQs)

        self._gradientCalculator = \
         TwoPopulationsIFEDMsSimpleGradientCalculator(a0=a0, a1=a1, a2=a2, 
                                                         dt=dt, dv=dv, 
                                                         reversedQs=reversedQs)
        self._llCalculator = TwoPopulationsLLCalculator()


    def optimize(self, eYs, iYs, wEI0, wIE0, eRho0, iRho0, eInputs, ysSigma2,
                       stepSize=1e-3, tol=1e-4, maxIter=1000):
        # y?s should be sampled at dtSaveRhos
        wEI = wEI0
        wIE = wIE0
        normGradient = float('Inf')
        ll = -float('Inf')
        iterNo = 0
        ws = np.empty((maxIter, 2))
        lls = np.empty(maxIter)
        ws[iterNo, 0] = wEI
        ws[iterNo, 1] = wIE
        lls[iterNo] = ll

        normGradients = np.empty(maxIter)
        while normGradient>tol and iterNo<maxIter:
            iterNo = iterNo + 1
            print("Iteration %d, wEI=%.02f, wIE=%.02f, |Grad|=%f, ll=%.02f" % \
                  (iterNo, wEI, wIE, normGradient, ll))
            sys.stdout.flush()

            eRhos, iRhos, eRs, iRs = \
             self._twoPIntegrator.integrate(eRho0=eRho0, iRho0=iRho0,
                                                   wEI=wEI, wIE=wIE, 
                                                   eInputs=eInputs)
            dWEI, dWIE = self._gradientCalculator.deriv(w12=wEI, w21=wIE,
                                                           y1s=eYs, y2s=iYs,
                                                           r1s=eRs, r2s=iRs,
                                                           rho1s=eRhos, 
                                                           rho2s=iRhos, 
                                                           eInputs=eInputs,
                                                           ysSigma2=ysSigma2)
            ll = self._llCalculator.calculateLL(eYs=eYs, iYs=iYs, eRs=eRs, 
                                                         iRs=iRs, 
                                                   sigma2=ysSigma2)
            normGradient = math.sqrt(dWEI**2+dWIE**2)
            wEI = wEI + stepSize * dWEI
            wIE = wIE + stepSize * dWIE

            ws[iterNo, 0] = wEI
            ws[iterNo, 1] = wIE
            lls[iterNo] = ll
            normGradients[iterNo] = normGradient

        if normGradient<tol:
            converged = True
        else:
            converged = False
        ws = ws[:iterNo+1,]
        lls = lls[:iterNo+1]
        normGradients = normGradients[:iterNo]
        return(wEI, wIE, converged, ws, lls, normGradients)

