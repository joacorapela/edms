
import sys
import numpy as np
# import matplotlib.pyplot as plt
import pdb

from IFSimulatorRWFWI import IFSimulatorRWFWI
# from myUtils import plotSpikeRates
# from myUtils import plotRhos

def main(argv):
    nNeurons = 900
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
#     n1 = 6.0
#     kappaMu = n1/210
#     kappaSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    nInputsPerNeuron = 10
    fracExcitatoryNeurons = 0.2

    mu = 0.50
    sigma = 0.1
    vs0 = np.random.normal(loc=mu, scale=sigma, size=nNeurons)
    ss0 = np.zeros(nNeurons)

    t0 = 0.0
    tf = 0.20
    dt = 1e-5
    dtSaveRhos = 1e-3
#     startTimePlotRhos = 0.05
    resultsFilename = 'results/sinusoidalRWFWINNeurons%dKMu%.04f.npz' % (nNeurons, kappaMu)
#     spikeRatesFigsFilename = 'figures/spikeRatesSinusoidalRWFWINNeurons%dKMu%.04f.eps' % (nNeurons, kappaMu)
#     rhosFigsFilename = 'figures/rhosSinusoidalRWFWINNeurons%dKM%.04f.eps' % (nNeurons, kappaMu)
#     resultsFilename = 'results/stepRWFWINNeurons%d.npz' % nNeurons
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRWFWINNeurons%d.eps' % nNeurons
#     rhosFigsFilename = 'figures/rhosStepRWFWINNeurons%d.eps' % nNeurons
#     averageWinTimeLength = 1e-3

    aIFSimulator = IFSimulatorRWFWI(nNeurons=nNeurons, leakage=leakage, 
                                     hMu=hMu, hSigma=hSigma,
                                     nInputsPerNeuron=nInputsPerNeuron,
                                     kappaMu=kappaMu, kappaSigma=kappaSigma,
                                     fracExcitatoryNeurons=
                                      fracExcitatoryNeurons)

    def stepInputMeanFrequency(t, sigma0=800):
        return(sigma0*np.ones(ts.size))

    def sinusoidalInputMeanFrequency(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    aIFSimulator.prepareToSimulate(t0=t0, tf=tf, dt=dt, nVSteps=nVSteps, 
                                          vs0=vs0, ss0=ss0, 
                                          eInputCurrent=sinusoidalInputMeanFrequency, 
                                          iInputCurrent=None)
    ts, rhos, spikeRates, saveRhosTimeDSFactor = \
     aIFSimulator.simulate(dtSaveRhos=dtSaveRhos)
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps

    np.savez(resultsFilename, spikeRates=spikeRates, ts=ts, vs=vs,
                              rhos=rhos,
                              saveRhosTimeDSFactor=saveRhosTimeDSFactor,
                              eInputCurrentHist=\
                               aIFSimulator.getEInputCurrentHist(),
                              eFeedbackCurrentHist=\
                               aIFSimulator.getEFeedbackCurrentHist(),
                              iInputCurrentHist=\
                               aIFSimulator.getIInputCurrentHist(),
                              iFeedbackCurrentHist=\
                               aIFSimulator.getIFeedbackCurrentHist())

#     plt.figure()
#     plotSpikeRates(ts, spikeRates, dt, averageWinTimeLength)
#     plt.savefig(spikeRatesFigsFilename)
#     startSamplePlotRhos = startTimePlotRhos/dt
#     plt.grid()
#     plt.figure()
#     plotRhos(vs, ts[startSamplePlotRhos:], rhos[:, startSamplePlotRhos:])
#     plt.savefig(rhosFigsFilename)

#     pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

