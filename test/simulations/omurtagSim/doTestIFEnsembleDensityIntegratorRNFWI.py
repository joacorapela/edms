
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from myUtils import plotSpikeRates, plotRhos
from IFEnsembleDensityIntegratorRNFWI import IFEnsembleDensityIntegratorRNFWI

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
#     n1 = 6.0
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    nInputsPerNeuron = 10
    rho0 = np.zeros(nVSteps)
    rho0[0] = 1.0*nVSteps
    t0 = 0.0
    tf = 1.0
    dt = 1e-5
    spikeRate0 = 0.0
    startTimePlotRhos = 0.05
    resultsFilename = 'results/sinusoidalRNFWIPopulationKMu%.4f.npz'% kappaMu
    spikeRatesFigsFilename = 'figures/spikeRatesSinusoidalRNFWIPopulationKMu%.4f.eps' % kappaMu
    rhosFigsFilename = 'figures/rhosSinusoidalRNFWIPopulationKMu%.4f.eps' % kappaMu
#     resultsFilename = 'results/stepRNFWIPopulation.npz'
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRNFWIPopulation.eps'
#     rhosFigsFilename = 'figures/rhosStepRNFWIPopulation.eps'

    def stepInputMeanFrequency(t, sigma0=800):
        return(sigma0)

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
    
    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRNFWI(nVSteps=nVSteps, leakage=leakage, 
                                                       hMu=hMu, hSigma=hSigma, 
                                                       kappaMu=kappaMu, 
                                                       kappaSigma=kappaSigma)
    ifEDIntegrator.prepareToIntegrate(rho0=rho0, spikeRate0=spikeRate0,
                                                 t0=t0, tf=tf, dt=dt, 
                                                 eInputCurrent=eStim,
                                                 iInputCurrent=iStim)
    ts, rhos, spikeRates = ifEDIntegrator.integrate()

    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    np.savez(resultsFilename, spikeRates=spikeRates, rhos=rhos, ts=ts, vs=vs,
                              eInputCurrentHist=\
                               ifEDIntegrator.getEInputCurrentHist(),
                              iInputCurrentHist=\
                               ifEDIntegrator.getIInputCurrentHist())
    averageWinTimeLength = 1e-3                  
    plt.figure()
    plotSpikeRates(ts=ts, spikeRates=spikeRates, dt=dt,
                                     averageWinTimeLength=averageWinTimeLength)
    plt.savefig(spikeRatesFigsFilename)
    startSamplePlotRhos = startTimePlotRhos/dt
    plt.figure()
    plotRhos(vs, ts[startSamplePlotRhos:], rhos[:, startSamplePlotRhos:])
    plt.savefig(rhosFigsFilename)

    pdb.set_trace()
    
if __name__ == "__main__":
    main(sys.argv)

