
import sys
import numpy as np
import math
import pdb

class TwoPopulationsIFEnsembleDensityIntegrator:

    def __init__(self, ifEDIntegrator1, ifEDIntegrator2):
        self._ifEDIntegrator1 = ifEDIntegrator1
        self._ifEDIntegrator2 = ifEDIntegrator2

    def prepareToIntegrate(self, t0, tf, dt, nVSteps, 
                                 edm1EExternalInput, edm2EExternalInput, 
                                 edm1IExternalInput, edm2IExternalInput):
        self._ifEDIntegrator1.prepareToIntegrate(t0=t0, 
                                                  tf=tf,
                                                  dt=dt,
                                                  eExternalInput=
                                                   edm1EExternalInput,
                                                  iExternalInput=
                                                   edm1IExternalInput)
        self._ifEDIntegrator2.prepareToIntegrate(t0=t0, 
                                                  tf=tf,
                                                  dt=dt,
                                                  eExternalInput=
                                                   edm2EExternalInput,
                                                  iExternalInput=
                                                   edm2IExternalInput)

    def setInitialValue(self, edm1Rho0, edm2Rho0):
        self._ifEDIntegrator1.setInitialValue(rho0=edm1Rho0)
        self._ifEDIntegrator2.setInitialValue(rho0=edm2Rho0)

    def integrate(self, dtSaveRhos):
        t0 = self._ifEDIntegrator1.getT0()
        tf = self._ifEDIntegrator1.getTf()
        dt = self._ifEDIntegrator1.getDt()
        nTSteps = round((tf-t0)/dt)
        if not math.isinf(dtSaveRhos):
            nTStepsSaveRhos = round((tf-t0)/dtSaveRhos)
            saveRhosTimeDSFactor = int(round(dtSaveRhos/dt))
            edm1Rhos = np.empty((self._ifEDIntegrator1.getRho0().size, 
                                  nTStepsSaveRhos))
            edm1Rhos[:] = np.nan
            edm2Rhos = np.empty((self._ifEDIntegrator2.getRho0().size, 
                                  nTStepsSaveRhos))
            edm2Rhos[:] = np.nan
            edm1Rhos[:, 0] = self._ifEDIntegrator1.getRho0()
            edm2Rhos[:, 0] = self._ifEDIntegrator2.getRho0()
        else:
            saveRhosTimeDSFactor = float("Inf")
            edm1Rhos = None
            edm2Rhos = None
        edm1Times = np.empty(nTSteps)
        edm1Times[:] = np.nan
        edm2Times = np.empty(nTSteps)
        edm2Times[:] = np.nan
        edm1Times[0] = t0
        edm2Times[0] = t0
        edm1SuccessfulIntegration = True
        edm2SuccessfulIntegration = True
        edm1T = t0
        edm2T = t0
        step = 0
        stepRho = 0

        while(edm1SuccessfulIntegration and edm2SuccessfulIntegration and 
                                     step<nTSteps-1):
            step = step+1
            if step%100==0:
                print("Processing time %.05f out of %.02f" % (edm1T, tf))
                sys.stdout.flush()

            edm1SuccessfulIntegration, edm1T, edm1Rho= \
             self._ifEDIntegrator1.integrateOneDeltaT(edm1T)
            edm1Times[step] = edm1T

            edm2SuccessfulIntegration, edm2T, edm2Rho= \
             self._ifEDIntegrator2.integrateOneDeltaT(edm2T)
            edm2Times[step] = edm2T

            if step%saveRhosTimeDSFactor==0:
                stepRho = stepRho+1
                edm1Rhos[:, stepRho] = edm1Rho
                edm2Rhos[:, stepRho] = edm2Rho

        return(edm1Times, edm1Rhos, self._ifEDIntegrator1.getSpikeRates(), 
                edm2Times, edm2Rhos,  self._ifEDIntegrator2.getSpikeRates(), 
                saveRhosTimeDSFactor)

