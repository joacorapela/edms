
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from plotEDMsResults import plotSpikeRatesAndCurrents, plotRhos
from IFEnsembleDensityIntegratorDDiffusionNFNI import IFEnsembleDensityIntegratorDDiffusionNFNI

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
#     rho0 = np.zeros(nVSteps)
#     rho0[0] = 1.0*nVSteps
    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    t0 = 0.0
    tf = 1.0
    dt = 1e-3
    spikeRate0 = 0.0
    startTimePlotRhos = 0.05
    resultsFilename = 'results/testLinearityRho1_rho.npz'
    spikeRatesFigsFilename = 'figures/spikeRatesTestLinearityRho1_rho.eps'

    sigma0 = 800
    b = 10
    freq = 4
    def sinusoidalInputMeanFrequency(t, sigma0=sigma0, b=b,
                                        omega=2*np.pi*freq):
        return(sigma0+b*np.sin(omega*t))

    ifEDIntegrator = \
     IFEnsembleDensityIntegratorDDiffusionNFNI(nVSteps=nVSteps, leakage=leakage,
                                                                hMu=hMu)
    ifEDIntegrator.prepareToIntegrate(t0=t0, tf=tf, dt=dt,
                                           eInputCurrent=sinusoidalInputMeanFrequency)
    ifEDIntegrator.setInitialValue(rho0=rho0)
    ts, rhos, spikeRates, _ = ifEDIntegrator.integrate(dtSaveRhos=dt)
    np.savez(resultsFilename, ts=ts, vs=vs, rhos=rhos, spikeRates=spikeRates)

    averageWinTimeLength = 1e-3                  
    plt.figure()
    plotSpikeRatesAndCurrents(ts, spikeRates, dt, averageWinTimeLength)
    plt.savefig(spikeRatesFigsFilename)
    startSamplePlotRhos = startTimePlotRhos/dt
#     plt.figure()
#     plotRhos(vs, ts[startSamplePlotRhos:], rhos[:, startSamplePlotRhos:])
#     plt.savefig(rhosFigsFilename)

if __name__ == "__main__":
    main(sys.argv)

