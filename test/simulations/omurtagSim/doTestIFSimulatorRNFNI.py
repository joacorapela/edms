
import sys
import numpy as np
# import matplotlib.pyplot as plt
import pdb

from IFSimulatorRNFNI import IFSimulatorRNFNI
# from myUtils import plotSpikeRates
# from myUtils import plotRhos

def main(argv):
    nNeurons = 9000
    vs0 = np.zeros(nNeurons)
    nVSteps = 210
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    leakage = 20
    vs0 = np.zeros(nNeurons)
    ss0 = np.zeros(nNeurons)
    t0 = 0.0
    tf = 1.0
    dt = 1e-5
    dtSaveRhos = 1e-3
#     startTimePlotRhos = 0.05
    resultsFilename = 'results/sinusoidalRNFNINNeurons%d.npz' % (nNeurons)
#     spikeRatesFigsFilename = 'figures/spikeRatesSinusoidalRNFNINNeurons%d.eps' % (nNeurons)
#     rhosFigsFilename = 'figures/rhosSinusoidalRNFNINNeurons%d.eps' % (nNeurons)
#     resultsFilename = 'results/stepRNFNINNeurons%d.npz' % nNeurons
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRNFNINNeurons%d.eps' % nNeurons
#     rhosFigsFilename = 'figures/rhosStepRNFNINNeurons%d.eps' % nNeurons
#     averageWinTimeLength = 1e-3

    def stepInputMeanFrequency(ts, sigma0=800):
        if(isinstance(ts, float)):
            return(sigma0)
        return(sigma0*np.ones(ts.size))

    def sinusoidalInputMeanFrequency(ts, sigma0=800, b=0.6, omega=8*np.pi):
        return(sigma0*(1+b*np.sin(omega*ts)))

    aIFSimulator = IFSimulatorRNFNI(nNeurons=nNeurons, leakage=leakage, 
                                     hMu=hMu, hSigma=hSigma)
    aIFSimulator.prepareToSimulate(t0=t0, tf=tf, dt=dt, nVSteps=nVSteps,
                                          vs0=vs0, ss0=ss0,
                                          eInputCurrent=sinusoidalInputMeanFrequency)
#                                           eInputCurrent=stepInputMeanFrequency)

    times, rhos, spikeRates, saveRhosTimeDSFactor = \
     aIFSimulator.simulate(dtSaveRhos=dtSaveRhos)

    np.savez(resultsFilename, times=times, saveRhosTimeDSFactor=saveRhosTimeDSFactor, rhos=rhos, spikeRates=spikeRates, eInputCurrentHist=aIFSimulator.getEInputCurrentHist())
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

