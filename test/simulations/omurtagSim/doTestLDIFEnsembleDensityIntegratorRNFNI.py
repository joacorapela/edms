
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
from myUtils import plotSpikeRates, plotRhos
from LDIFEnsembleDensityIntegratorRNFNI \
 import LDIFEnsembleDensityIntegratorRNFNI

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
    tf = 1.00
    dt = 1e-5
    spikeRate0 = 0.0
    startTimePlotRhos = 0.05
    eigenReposFilename = "results/anEigenResposForOneStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571StimFrom300.000000StimTo1300.000000StimStep20.000000NEigen17.pickle"
    resultsFilename = "results/sinusoidalLDRNFNIPopulation.npz"
    spikeRatesFigsFilename = "figures/spikeRatesSinusoidalLDRNFNIPopulation.eps"
#     sriLDCoefsColFigsFilename = "figures/sriLDCoefsColSinusoidalRNFNIPopulation.eps"
#     resultsFilename = "results/stepLDRNFNIPopulation.npz"
#     spikeRatesFigsFilename = "figures/spikeRatesStepLDRNFNIPopulation.eps"
#     sriLDCoefsColFigsFilename = "figures/rhosStepLDRNFNIPopulation.eps"
    nEigen = 3

    sigma0 = 800
    b = 0.6
    freq = 4
    def sinusoidalInputMeanFrequency(t, sigma0=sigma0, b=b, 
                                        omega=2*np.pi*freq):
        return(sigma0*(1+b*np.sin(omega*t)))

    with open(eigenReposFilename, "rb") as f:
        eigenRepos = pickle.load(f)
    ldIFEDIntegrator = \
     LDIFEnsembleDensityIntegratorRNFNI(nVSteps=nVSteps, leakage=leakage,
                                                         hMu=hMu,
                                                         hSigma=hSigma, 
                                                         nEigen=nEigen,
                                                         eigenRepos=eigenRepos)
    ldIFEDIntegrator.prepareToIntegrate(t0=t0, tf=tf, dt=dt,
                                               spikeRate0=spikeRate0,
                                               eInputCurrent=sinusoidalInputMeanFrequency)
#                                                eInputCurrent=stepInputMeanFrequency)
    ldIFEDIntegrator.setInitialValue(rho0=rho0)
    ts, sriLDCoefsCol, spikeRates = ldIFEDIntegrator.integrate()
#     np.savez(resultsFilename, ts=ts, vs=vs, sriLDCoefsCol=sriLDCoefsCol, spikeRates=spikeRates)

    averageWinTimeLength = 1e-3                  
    plt.figure()
    plotSpikeRates(ts, spikeRates, dt, averageWinTimeLength)
    plt.savefig(spikeRatesFigsFilename)
#     startSamplePlotRhos = startTimePlotRhos/dt
#     plt.figure()
#     plotRhos(vs, ts[startSamplePlotRhos:], sriLDCoefsCol[:, startSamplePlotRhos:])
#     plt.savefig(sriLDCoefsColFigsFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

