
import sys
import numpy as np
import math
import pdb

class TwoPopulationsLDIFEnsembleDensityIntegrator:

    def __init__(self, ifEDIntegrator1, ifEDIntegrator2):
        self._ifEDIntegrator1 = ifEDIntegrator1
        self._ifEDIntegrator2 = ifEDIntegrator2

    def prepareToIntegrate(self, nVSteps, t0, tf, dt,
                                 edm1EInputCurrent, edm2EInputCurrent, 
                                 edm1IInputCurrent, edm2IInputCurrent):
        self._ifEDIntegrator1.prepareToIntegrate(t0=t0, 
                                                  tf=tf,
                                                  dt=dt,
                                                  eInputCurrent=
                                                   edm1EInputCurrent,
                                                  iInputCurrent=
                                                   edm1IInputCurrent)
        self._ifEDIntegrator2.prepareToIntegrate(t0=t0, 
                                                  tf=tf,
                                                  dt=dt,
                                                  eInputCurrent=
                                                   edm2EInputCurrent,
                                                  iInputCurrent=
                                                   edm2IInputCurrent)
    def setInitialValue(self, edm1Rho0, edm2Rho0):
        self._ifEDIntegrator1.setInitialValue(rho0=edm1Rho0)
        self._ifEDIntegrator2.setInitialValue(rho0=edm2Rho0)

    def integrate(self, dtSaveLDCoefs):
        t0 = self._ifEDIntegrator1.getT0()
        tf = self._ifEDIntegrator1.getTf()
        dt = self._ifEDIntegrator1.getDt()
        nEigen = self._ifEDIntegrator1.getNEigen()

        nTSteps = round((tf-t0)/dt)
        nTStepsSaveLDCoefs = round((tf-t0)/dtSaveLDCoefs)
        saveLDCoefsTimesDSFactor = round(dtSaveLDCoefs/dt)
        edm1Times = np.empty(nTSteps+1)
        edm1Times[:] = np.nan
        edm2Times = np.empty(nTSteps+1)
        edm2Times[:] = np.nan
        edm1SRILDCoefsCol = np.empty((2*nEigen, nTStepsSaveLDCoefs+1))
        edm1SRILDCoefsCol[:] = np.nan
        edm2SRILDCoefsCol = np.empty((2*nEigen, nTStepsSaveLDCoefs+1))
        edm2SRILDCoefsCol[:] = np.nan
        edm1Times[0] = t0
        edm2Times[0] = t0
        edm1SRILDCoefsCol[:, 0] = self._ifEDIntegrator1.getSRILDCoefs()
        edm2SRILDCoefsCol[:, 0] = self._ifEDIntegrator2.getSRILDCoefs()
        edm1SuccessfulIntegration = True
        edm2SuccessfulIntegration = True
        edm1T = t0
        edm2T = t0
        step = 0
        stepLDCoefs = 0
        edm1SRILDCoefs = self._ifEDIntegrator1.getSRILDCoefs()
        edm2SRILDCoefs = self._ifEDIntegrator2.getSRILDCoefs()

        while(edm1SuccessfulIntegration and edm2SuccessfulIntegration and 
                                     step<nTSteps):
            step = step+1
            if step%100==0:
                print("Processing time %.05f out of %.02f (spike rates=(%f, %f))" %
                      (edm1T, tf, edm1SpikeRate, edm2SpikeRate))
                sys.stdout.flush()

            edm1SuccessfulIntegration, edm1T, edm1SRILDCoefs, edm1SpikeRate = \
             self._ifEDIntegrator1.integrateOneDeltaT(t=edm1T,
                                                       sriLDCoefs=
                                                        edm1SRILDCoefs)
            edm1Times[step] = edm1T

            edm2SuccessfulIntegration, edm2T, edm2SRILDCoefs, edm2SpikeRate = \
             self._ifEDIntegrator2.integrateOneDeltaT(t=edm2T, 
                                                       sriLDCoefs=
                                                        edm2SRILDCoefs)
            edm2Times[step] = edm2T

            if step%saveLDCoefsTimesDSFactor==0:
                stepLDCoefs = stepLDCoefs+1
                edm1SRILDCoefsCol[:, stepLDCoefs] = edm1SRILDCoefs
                edm2SRILDCoefsCol[:, stepLDCoefs] = edm2SRILDCoefs
        return(edm1Times, edm1SRILDCoefsCol, 
                          self._ifEDIntegrator1.getSpikeRates(), 
               edm2Times, edm2SRILDCoefsCol, 
                          self._ifEDIntegrator2.getSpikeRates(),
               saveLDCoefsTimesDSFactor)

