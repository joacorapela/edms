
import sys
import numpy as np
# import matplotlib.pyplot as plt
import pdb

from IFSimulatorRNFWI import IFSimulatorRNFWI
# from myUtils import plotSpikeRates
# from myUtils import plotRhos

def main(argv):
    nNeurons = 9000
    nVSteps = 210
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
#     n1 = 6.0
#     kappaMu = n1/210
#     kappaSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    leakage = 20
    vs0 = np.zeros(nNeurons)
    ss0 = np.zeros(nNeurons)
    t0 = 0.0
    tf = 1.0
    dt = 1e-5
#     startTimePlotRhos = 0.05
    resultsFilename = '/tmp/sinusoidalRNFWINNeurons%dKMu%.04f.npz' % (nNeurons, kappaMu)
#     spikeRatesFigsFilename = 'figures/spikeRatesSinusoidalRNFWINNeurons%dKMu%.04f.eps' % (nNeurons, kappaMu)
#     rhosFigsFilename = 'figures/rhosSinusoidalRNFWINNeurons%dKM%.04f.eps' % (nNeurons, kappaMu)
#     resultsFilename = 'results/stepRNFWINNeurons%d.npz' % nNeurons
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRNFWINNeurons%d.eps' % nNeurons
#     rhosFigsFilename = 'figures/rhosStepRNFWINNeurons%d.eps' % nNeurons
#     averageWinTimeLength = 1e-3

    aIFSimulator = IFSimulatorRNFWI(nNeurons=nNeurons, leakage=leakage, 
                                     hMu=hMu, hSigma=hSigma,
                                     kappaMu=kappaMu, kappaSigma=kappaSigma)

    def stepInputMeanFrequency(ts, sigma0=800):
        if(isinstance(ts, float)):
            return(sigma0)
        return(sigma0*np.ones(ts.size))

    sigma0E = 800
    freqE = 4
    sigma0I = 200
    freqI = 1

    b = 0.6
    def sinusoidalInputMeanFrequency(t, sigma0, omega, b=b, phase=-np.pi/2):
        return(sigma0*(1+b*np.sin(omega*t+phase)))

    eStim = lambda t: sinusoidalInputMeanFrequency(t=t, sigma0=sigma0E, 
                                                        omega=2*np.pi*freqE)
    iStim = lambda t: sinusoidalInputMeanFrequency(t=t, sigma0=sigma0I,
                                                        omega=2*np.pi*freqI)
    
    aIFSimulator.prepareToSimulate(t0=t0, tf=tf, dt=dt, nVSteps=nVSteps, 
                                          vs0=vs0, ss0=ss0, 
                                          eInputCurrent=eStim,
                                          iInputCurrent=iStim)
#     ts, rhos, spikeRates = aIFSimulator.simulate()
    ts, spikeRates = aIFSimulator.simulate()
#     vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
#     np.savez(resultsFilename, ts=ts, vs=vs, rhos=rhos, spikeRates=spikeRates,
#                               eInputCurrentHist=\
#                                aIFSimulator.getEInputCurrentHist(),
#                               eFeedbackCurrentHist=\
#                                aIFSimulator.getEFeedbackCurrentHist(),
#                               iInputCurrentHist=\
#                                aIFSimulator.getIInputCurrentHist(),
#                               iFeedbackCurrentHist=\
#                                aIFSimulator.getIFeedbackCurrentHist())

    np.savez(resultsFilename, ts=ts, spikeRates=spikeRates,
                              eInputCurrentHist=\
                               aIFSimulator.getEInputCurrentHist(),
                              iInputCurrentHist=\
                               aIFSimulator.getIInputCurrentHist())

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

