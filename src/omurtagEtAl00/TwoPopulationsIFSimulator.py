
import sys
import numpy as np
import math
import pdb

class TwoPopulationsIFSimulator:

    def __init__(self, ifSimulator1, ifSimulator2):
        self._ifSimulator1 = ifSimulator1
        self._ifSimulator2 = ifSimulator2

    def prepareToSimulate(self, t0, tf, dt, nVSteps, sim1Vs0, sim1Ss0, sim2Vs0, sim2Ss0,
                                sim1EInputCurrent, sim1IInputCurrent,
                                sim2EInputCurrent, sim2IInputCurrent):
        self._ifSimulator1.prepareToSimulate(vs0=sim1Vs0, 
                                              ss0=sim1Ss0,
                                              t0=t0, tf=tf, dt=dt, 
                                              nVSteps=nVSteps,
                                              eInputCurrent=sim1EInputCurrent,
                                              iInputCurrent=sim1IInputCurrent)
        self._ifSimulator2.prepareToSimulate(vs0=sim2Vs0, 
                                              ss0=sim2Ss0,
                                              t0=t0, tf=tf, dt=dt, 
                                              nVSteps=nVSteps,
                                              eInputCurrent=sim2EInputCurrent,
                                              iInputCurrent=sim2IInputCurrent)

    def simulate(self, dtSaveRhos):
        t0 = self._ifSimulator1.getT0()
        tf = self._ifSimulator1.getTf()
        dt = self._ifSimulator1.getDt()
        nVSteps = self._ifSimulator1.getNVSteps()

        vsiSim1 = np.copy(self._ifSimulator1.getVs0())
        ssiSim1 =  np.copy(self._ifSimulator1.getSs0())
        vsiSim2 = np.copy(self._ifSimulator2.getVs0())
        ssiSim2 =  np.copy(self._ifSimulator2.getSs0())

        nTSteps = round((tf-t0)/dt)
        nTStepsSaveRhos = round((tf-t0)/dtSaveRhos)
        saveRhosTimeDSFactor = int(round(dtSaveRhos/dt))
        times = np.empty(nTSteps+1)
        times[:] = np.nan
        rhosSim1 = np.empty((nVSteps, nTStepsSaveRhos+1))
        rhosSim1[:] = np.nan
        rhosSim2 = np.empty((nVSteps, nTStepsSaveRhos+1))
        rhosSim2[:] = np.nan
        times[0] = t0
        rhosSim1[:, 0] = np.histogram(self._ifSimulator1.getVs0(), bins=nVSteps, 
                                                                   range=(0, 1), normed=True)[0]
        rhosSim2[:, 0] = np.histogram(self._ifSimulator2.getVs0(), bins=nVSteps, 
                                                                   range=(0, 1), normed=True)[0]

        t = t0
        step = 0
        stepRho = 0
        sim1SR = 0.0
        sim2SR = 0.0
        while step<nTSteps:
            step = step+1
            t = t + dt
            if step%100==0:
                print("Processing time %.05f out of %.02f (spike rates=(%f, %f))" %
                      (t, tf, sim1SR, sim2SR))
                sys.stdout.flush()
            vsiSim1, ssiSim1, sim1SR = self._ifSimulator1.simulateOneStep(t=t, vsi=vsiSim1, ssi=ssiSim1)
            vsiSim2, ssiSim2, sim2SR = self._ifSimulator2.simulateOneStep(t=t, vsi=vsiSim2, ssi=ssiSim2)
            times[step] = t
            if step%saveRhosTimeDSFactor==0:
                stepRho = stepRho+1
                rhosSim1[:, stepRho] = np.histogram(vsiSim1, bins=nVSteps, 
                                                             range=(0, 1), normed=True)[0]
                rhosSim2[:, stepRho] = np.histogram(vsiSim2, bins=nVSteps, 
                                                             range=(0, 1), normed=True)[0]
        return(times, rhosSim1, rhosSim2, 
                      self._ifSimulator1.getSpikeRates(), 
                      self._ifSimulator2.getSpikeRates(),
                      saveRhosTimeDSFactor)

