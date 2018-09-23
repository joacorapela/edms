
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from myUtils import plotSpikeRates, plotRhos
from IFEnsembleDensityIntegratorRWFNI import IFEnsembleDensityIntegratorRWFNI

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
#     n1 = 6.0
    nInputsPerNeuron = 10
    rho0 = np.zeros(nVSteps)
    rho0[0] = 1.0*nVSteps
    t0 = 0.0
    tf = 1.0
    dt = 1e-5
    spikeRate0 = 0.0
    startTimePlotRhos = 0.05
    resultsFilename = 'results/sinusoidalRWFNIPopulation.npz'
    spikeRatesFigsFilename = 'figures/spikeRatesSinusoidalRWFNIPopulation.eps'
    rhosFigsFilename = 'figures/rhosSinusoidalRWFNIPopulation.eps'
#     resultsFilename = 'results/stepRWFNIPopulation.npz'
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRWFNIPopulation.eps'
#     rhosFigsFilename = 'figures/rhosStepRWFNIPopulation.eps'

    def stepInputMeanFrequency(ts, sigma0=800):
        if(isinstance(ts, float)):
            return(sigma0)
        return(sigma0*np.ones(ts.size))

    def sinusoidalInputMeanFrequency(ts, sigma0=800, b=0.6, omega=8*np.pi):
        return(sigma0*(1+b*np.sin(omega*ts)))

    ensembleInt = \
     IFEnsembleDensityIntegratorRWFNI(nVSteps, leakage, hMu, hSigma, 
                                               nInputsPerNeuron)
    ensembleInt.prepareToIntegrate(rho0=rho0, t0=t0, tf=tf, dt=dt,
                                              spikeRate0=spikeRate0,
                                              eInputCurrent=sinusoidalInputMeanFrequency)
    ts, rhos, spikeRates = ensembleInt.integrate()
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    np.savez(resultsFilename, spikeRates=spikeRates, rhos=rhos, ts=ts, vs=vs)
    averageWinTimeLength = 1e-3                  
    plt.figure()
    plotSpikeRates(ts, spikeRates, dt, averageWinTimeLength)
    plt.savefig(spikeRatesFigsFilename)
    startSamplePlotRhos = startTimePlotRhos/dt
    plt.figure()
    plotRhos(vs, ts[startSamplePlotRhos:], rhos[:, startSamplePlotRhos:])
    plt.savefig(rhosFigsFilename)

    pdb.set_trace()
    
if __name__ == "__main__":
    main(sys.argv)

