
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from plotEDMsResults import plotSpikeRatesAndCurrents, plotRhos
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
#     rho0 = np.zeros(nVSteps)
#     rho0[0] = 1.0*nVSteps
    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    t0 = 0.0
    tf = 1.0
    dt = 1e-5
    dtSaveRhos = 1e-3
    ylim = (0, 50)
    resultsFilename = 'results/zeroRNFNIPopulationB.npz'
    spikeRatesFigsFilename = 'figures/spikeRatesZeroRNFNIPopulationB.eps'
#     resultsFilename = 'results/sinusoidalRNFNIPopulationB.npz'
#     spikeRatesFigsFilename = 'figures/spikeRatesSinusoidalRNFNIPopulationB.eps'
#     rhosFigsFilename = 'figures/rhosSinusoidalRNFNIPopulationB.eps'
#     resultsFilename = 'results/stepRNFNIPopulationB.npz'
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRNFNIPopulationB.eps'
#     rhosFigsFilename = 'figures/rhosStepRNFNIPopulationB.eps'

    sigma0 = 0
#     sigma0 = 800

    def stepInputMeanFrequency(t, sigma0=sigma0):
        return(sigma0)

    b = 0.6
    freq = 4
    def sinusoidalInputMeanFrequency(t, sigma0=sigma0, b=b,
                                        omega=2*np.pi*freq):
        return(sigma0*(1+b*np.sin(omega*t)))

    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRNFNI(nVSteps=nVSteps, leakage=leakage, 
                                                       hMu=hMu, hSigma=hSigma)
    ifEDIntegrator.prepareToIntegrate(t0=t0, tf=tf, dt=dt,
#                                            eInputCurrent=sinusoidalInputMeanFrequency)
                                             eInputCurrent=stepInputMeanFrequency)
    ifEDIntegrator.setInitialValue(rho0=rho0)
    times, rhos, spikeRates, saveRhosTimeDSFactor = ifEDIntegrator.\
                                                     integrate(dtSaveRhos=dtSaveRhos)
    np.savez(resultsFilename, times=times,
                              saveRhosTimeDSFactor=saveRhosTimeDSFactor, vs=vs, 
                              rhos=rhos, spikeRates=spikeRates)

    averageWinTimeLength = 1e-5
    plt.figure()
    plotSpikeRatesAndCurrents(times, spikeRates, dt, averageWinTimeLength,
                                     ylimAx1=ylim)
    plt.savefig(spikeRatesFigsFilename)
#     plt.figure()
#     plotRhos(vs, times[startSamplePlotRhos:], rhos[:, startSamplePlotRhos:])
#     plt.savefig(rhosFigsFilename)

if __name__ == "__main__":
    main(sys.argv)

