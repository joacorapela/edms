
import sys
import numpy as np
import math
import pdb
from IFSimulatorRWFWI import IFSimulatorRWFWI
from TwoPopulationsIFSimulator import TwoPopulationsIFSimulator

def main(argv):
    nNeurons = 900
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    eNInputsPerNeuron = 10
    iNInputsPerNeuron = 10
    eFracExcitatoryNeurons = 0.2
    iFracExcitatoryNeurons = 0.2

    mu = 0.50
    sigma = 0.1
    vs0 = np.random.normal(loc=mu, scale=sigma, size=nNeurons)
    sim1Vs0 = vs0
    sim1Ss0 = np.zeros(nNeurons)
    sim2Vs0 = vs0
    sim2Ss0 = np.zeros(nNeurons)

    t0 = 0.0
    tf = 1.2
    dt = 1e-5
    dtSaveRhos = 1e-3
    resultsFilename = 'results/twoPSinusoidalRWFWINNeurons%d.npz' % nNeurons

    ifSimulator1 = IFSimulatorRWFWI(nNeurons=nNeurons, leakage=leakage, 
                                     hMu=hMu, hSigma=hSigma,
                                     nInputsPerNeuron=eNInputsPerNeuron,
                                     kappaMu=kappaMu, kappaSigma=kappaSigma,
                                     fracExcitatoryNeurons=
                                      eFracExcitatoryNeurons)
    ifSimulator2 = IFSimulatorRWFWI(nNeurons=nNeurons, leakage=leakage, 
                                     hMu=hMu, hSigma=hSigma,
                                     nInputsPerNeuron=iNInputsPerNeuron,
                                     kappaMu=kappaMu, kappaSigma=kappaSigma,
                                     fracExcitatoryNeurons=
                                      iFracExcitatoryNeurons)

    def sinusoidalInputMeanFrequency(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    def eSigmaForE(t):
        return(sinusoidalInputMeanFrequency(t))

    def iSigmaForE(t, w=15):
        r = ifSimulator2.getSpikeRate(t=t-dt)
        return(w*r)

    def eSigmaForI(t, w=50):
        r = ifSimulator1.getSpikeRate(t=t-dt)
        return(w*r)

    twoPSimulator = TwoPopulationsIFSimulator(ifSimulator1=ifSimulator1, 
                                               ifSimulator2=ifSimulator2)
    twoPSimulator.prepareToSimulate(t0=t0, 
                                     tf=tf, 
                                     dt=dt, 
                                     nVSteps=nVSteps,
                                     sim1Vs0=sim1Vs0,
                                     sim1Ss0=sim1Ss0,
                                     sim2Vs0=sim2Vs0,
                                     sim2Ss0=sim2Ss0,
                                     sim1EInputCurrent=eSigmaForE,
                                     sim1IInputCurrent=iSigmaForE,
                                     sim2EInputCurrent=eSigmaForI,
                                     sim2IInputCurrent=None)
    times, eRhos, iRhos, eSpikeRates, iSpikeRates, saveRhosTimeDSFactor = \
     twoPSimulator.simulate(dtSaveRhos=dtSaveRhos)

    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    np.savez(resultsFilename, vs=vs, 
                              eTimes=times, 
                              eRhos=eRhos,
                              eSpikeRates=eSpikeRates, 
                              iTimes=times, 
                              iRhos=iRhos,
                              saveRhosTimeDSFactor=saveRhosTimeDSFactor,
                              iSpikeRates=iSpikeRates,
                              eEInputCurrentHist=\
                               ifSimulator1.getEInputCurrentHist(),
                              eEFeedbackCurrentHist=\
                               ifSimulator1.getEFeedbackCurrentHist(),
                              eIInputCurrentHist=\
                               ifSimulator1.getIInputCurrentHist(),
                              eIFeedbackCurrentHist=\
                               ifSimulator1.getIFeedbackCurrentHist(),
                              iEInputCurrentHist=\
                               ifSimulator2.getEInputCurrentHist(),
                              iEFeedbackCurrentHist=\
                               ifSimulator2.getEFeedbackCurrentHist(),
                              iIInputCurrentHist=\
                               ifSimulator2.getIInputCurrentHist(),
                              iIFeedbackCurrentHist=\
                               ifSimulator2.getIFeedbackCurrentHist())


if __name__ == "__main__":
    main(sys.argv)

