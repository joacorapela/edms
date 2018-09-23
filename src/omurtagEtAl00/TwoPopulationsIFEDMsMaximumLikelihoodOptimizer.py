
import math
import numpy as np
import pdb
from TwoPopulationsIFEnsembleDensityIntegrator import TwoPopulationsIFEnsembleDensityIntegrator
from TwoPopulationsIFEDMsGradientCalculator import TwoPopulationsIFEDMsGradientCalculator

class TwoPopulationsIFEDMsMaximumLikelihoodOptimizer:

    def __init__(self, ifEDIntegrator1, ifEDIntegrator2, sigma0, 
                       edm1Rho0, edm2Rho0, t0, tf, dt, dv, dtSaveRhos, 
                       nVSteps):
        self._sigma0 = sigma0
        self._edm1Rho0 = edm1Rho0
        self._edm2Rho0 = edm2Rho0
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._dtSaveRhos = dtSaveRhos
        self._nVSteps = nVSteps
        self._twoPIFEDIntegrator = TwoPopulationsIFEnsembleDensityIntegrator(
                                    ifEDIntegrator1=ifEDIntegrator1,
                                    ifEDIntegrator2=ifEDIntegrator2)
        self._gradientCalculator = TwoPopulationsIFEDMsGradientCalculator( \
                                    a0=ifEDIntegrator1._a0, 
                                    a1=ifEDIntegrator1._a1, 
                                    a2=ifEDIntegrator1._a2, 
                                    sigma0 = sigma0,
                                    dt=dtSaveRhos, 
                                    dv=dv,
                                    reversedQs=ifEDIntegrator1._reversedQs)

    def optimize(self, y1s, y2s, w120, w210, stepSize=1e-3, tol=1e-4, 
                       maxIter=1000):
        # y?s should be sampled at dtSaveRhos
        w12 = w120
        w21 = w210
        normGradient = float('Inf')
        iterNo = 0
        ws = np.empty((maxIter, 2))
        while normGradient>tol and iterNo<maxIter:
            print("Iteration %d" % iterNo)
            # r?s should be sampled at dtSaveRhos
            times1, r1s, rho1s, times2, r2s, rho2s = \
             self._integrateEDMs(w12=w12, w21=w21)
            dW12, dW21 = self._gradientCalculator.\
                              deriv(w12=w12, w21=w21, y1s=y1s, y2s=y2s,
                                             times1=times1, times2=times2, 
                                             r1s=r1s, r2s=r2s, 
                                             rho1s=rho1s, rho2s=rho2s)
            normGradient = math.sqrt(dW12**2+dW21**2)
            w12 = w12 + stepSize * dW12
            w21 = w21 + stepSize * dW21
            ws[iterNo, 0] = w12
            ws[iterNo, 1] = w21
            iterNo = iterNo + 1
            pdb.set_trace()
        if mse<tol:
            converged = True
        else:
            converged = False
        ws = ws[:iterNo,]
        mses = mses[:iterNo]
        return(w12, w21, converged, ws, mses)

    def _integrateEDMs(self, w12, w21):
        def edm1ISigma(t, w=w21):
            r = self._twoPIFEDIntegrator._ifEDIntegrator2.\
                 getSpikeRate(t=t-self._dt)
            return(w*r)

        def edm2ESigma(t, w=w12):
            r = self._twoPIFEDIntegrator._ifEDIntegrator1.\
                 getSpikeRate(t=t-self._dt)
            return(w*r)

        self._twoPIFEDIntegrator.prepareToIntegrate(
                                  t0=self._t0,
                                  tf=self._tf,
                                  dt=self._dt,
                                  nVSteps=self._nVSteps,
                                  edm1EInputCurrent=self._sigma0,
                                  edm1IInputCurrent=edm1ISigma,
                                  edm2EInputCurrent=edm2ESigma,
                                  edm2IInputCurrent=None)
        self._twoPIFEDIntegrator.setInitialValue(edm1Rho0=self._edm1Rho0,
                                                  edm2Rho0=self._edm2Rho0)
        edm1Times, edm1Rhos, edm1SpikeRates, \
        edm2Times, edm2Rhos, edm2SpikeRates, \
        saveRhosTimeDSFactor = self._twoPIFEDIntegrator.\
                                integrate(dtSaveRhos=self._dtSaveRhos)
        return(edm1Times[::saveRhosTimeDSFactor], 
                edm1SpikeRates[::saveRhosTimeDSFactor], 
                edm1Rhos,
                edm2Times[::saveRhosTimeDSFactor], 
                edm2SpikeRates[::saveRhosTimeDSFactor], 
                edm2Rhos)
