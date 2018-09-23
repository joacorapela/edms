
import sys
import numpy as np
# import matplotlib.pyplot as plt
import pdb

from IFSimulatorRWFNI import IFSimulatorRWFNI
# from myUtils import plotSpikeRates
# from myUtils import plotRhos

def main(argv):
    nNeurons = 90
    vs0 = np.zeros(nNeurons)
    nVSteps = 210
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    leakage = 20
    nInputsPerNeuron = 10
    dt = 1e-5
    simDurationSecs = 0.2
    startTimePlotRhos = 0.05
    resultsFilename = 'results/sinusoidalRWFNINNeurons%d.npz' % (nNeurons)
#     spikeRatesFigsFilename = 'figures/spikeRatesSinusoidalRWFNINNeurons%d.eps' % (nNeurons)
#     rhosFigsFilename = 'figures/rhosSinusoidalRWFNINNeurons%d.eps' % (nNeurons)
#     resultsFilename = 'results/stepRWFNINNeurons%d.npz' % nNeurons
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRWFNINNeurons%d.eps' % nNeurons
#     rhosFigsFilename = 'figures/rhosStepRWFNINNeurons%d.eps' % nNeurons
    averageWinTimeLength = 1e-3

    def stepInputMeanFrequency(ts, sigma0=800):
        if(isinstance(ts, float)):
            return(sigma0)
        return(sigma0*np.ones(ts.size))

    def sinusoidalInputMeanFrequency(ts, sigma0=800, b=0.6, omega=8*np.pi):
        return(sigma0*(1+b*np.sin(omega*ts)))

    aIFSimulator = IFSimulatorRWFNI(nNeurons=nNeurons, leakage=leakage, 
                                     hMu=hMu, hSigma=hSigma,
                                     eInputCurrent=sinusoidalInputMeanFrequency,
                                     nInputsPerNeuron=nInputsPerNeuron)
    nTSteps = round(simDurationSecs/dt)
    aIFSimulator.prepareToSimulate(nTSteps=nTSteps, nVSteps=nVSteps)
    aIFSimulator.simulate(vs0, dt)
    probSpikePerBin = aIFSimulator.getProbSpikePerBin()
    rhos =aIFSimulator.getRhos()
    spikeRates = probSpikePerBin/dt

    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    ts = np.linspace(dt, spikeRates.size*dt, spikeRates.size)
    np.savez(resultsFilename, spikeRates=spikeRates, rhos=rhos, ts=ts, vs=vs)

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

