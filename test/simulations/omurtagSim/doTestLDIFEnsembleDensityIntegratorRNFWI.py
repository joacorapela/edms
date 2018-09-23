
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
from myUtils import plotSpikeRates, plotRhos
from LDIFEnsembleDensityIntegratorRNFWI \
 import LDIFEnsembleDensityIntegratorRNFWI

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    dv = 1.0/nVSteps

    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    t0 = 0.0
    tf = 1.00
    dt = 1e-5
    spikeRate0 = 0
    eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom300.000000EStimTo1500.000000EStimStep20.000000IStimFrom80.000000IStimTo320.000000IStimStep20.000000NEigen17.pickle"
    nEigen = 17
    spikeRatesFigsFilename = "figures/spikeRatesSinusoidalLDRNFWIPopulationNEigen%02d.eps" % (nEigen)

    sigma0E = 800
    freqE = 4
    sigma0I = 200
    freqI = 1

    b = 0.6
    def sinusoidalInputMeanFrequency(t, sigma0, omega, b=b, phase=-np.pi/2):
        return(sigma0*(1+b*np.sin(omega*t+phase)))

    def constantInputMeanFrequency(t, sigma0):
        return(sigma0)

    linearStimSlope = 1
    def linearInputMeanFrequency(t, sigma0, slope=linearStimSlope):
        return(sigma0+t*linearStimSlope)

    eStim = lambda t: sinusoidalInputMeanFrequency(t=t, sigma0=sigma0E,
                                                        omega=2*np.pi*freqE)
    iStim = lambda t: sinusoidalInputMeanFrequency(t=t, sigma0=sigma0I,
                                                        omega=2*np.pi*freqI)

    with open(eigenReposFilename, "rb") as f:
        eigenRepos = pickle.load(f)
    ldIFEDIntegrator = \
     LDIFEnsembleDensityIntegratorRNFWI(nVSteps=nVSteps, leakage=leakage,
                                                         hMu=hMu,
                                                         hSigma=hSigma, 
                                                         kappaMu=kappaMu,
                                                         kappaSigma=kappaSigma,
                                                         nEigen=nEigen,
                                                         eigenRepos=eigenRepos)
    ldIFEDIntegrator.prepareToIntegrate(t0=t0, tf=tf, dt=dt,
                                               spikeRate0=spikeRate0,
                                               eInputCurrent=eStim,
                                               iInputCurrent=iStim)
    ldIFEDIntegrator.setInitialValue(rho0=rho0)
    ts, sriLDCoefsCol, spikeRates = ldIFEDIntegrator.integrate()
#     np.savez(resultsFilename, ts=ts, vs=vs, sriLDCoefsCol=sriLDCoefsCol, spikeRates=spikeRates)

    averageWinTimeLength = 1e-3                  
    plt.figure()
    plotSpikeRates(ts, spikeRates, dt, averageWinTimeLength)
    plt.savefig(spikeRatesFigsFilename)
#     plt.figure()
#     plotRhos(vs, ts[startSamplePlotRhos:], sriLDCoefsCol[:, startSamplePlotRhos:])
#     plt.savefig(sriLDCoefsColFigsFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

