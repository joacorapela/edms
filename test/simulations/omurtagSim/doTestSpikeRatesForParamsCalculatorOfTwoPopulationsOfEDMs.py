
import sys
import numpy as np
import math
import pdb
from IFEnsembleDensityIntegratorRNFWI import IFEnsembleDensityIntegratorRNFWI 
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI 
from TwoPopulationsIFEnsembleDensityIntegrator import TwoPopulationsIFEnsembleDensityIntegrator
from SpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs import \
 SpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs

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
    eEDMRho0 = rho0
    iEDMRho0 = rho0
    t0 = 0.0
#     tf = 1.0
    tf = 0.25
    dt = 1e-5
    eEDMIWsStart = 0
    eEDMIWsStop = 50
    eEDMIWsStep = 5
    iEDMIWsStart = 0
    iEDMIWsStop = 100
    iEDMIWsStep = 10
    eEDMEWs = [None]
    eEDMIWs = np.arange(start=eEDMIWsStart, stop=eEDMIWsStop, step=eEDMIWsStep)
    iEDMEWs = np.arange(start=iEDMIWsStart, stop=iEDMIWsStop, step=iEDMIWsStep)
    iEDMIWs = [None]

    resultsFilename = "results/spikeRatesForParamsOfTwoPopulationsT0%.02fTf%.02fDt%fEEDMIWs%d-%d-%dIEDMEWs%d-%d-%d.npz" % (t0, tf, dt, eEDMIWsStart, eEDMIWsStop, eEDMIWsStep, iEDMIWsStart, iEDMIWsStop, iEDMIWsStep)

    def generateParams(eEDMEWs, eEDMIWs, iEDMEWs, iEDMIWs):
        params = []

        for i1E in xrange(len(eEDMEWs)):
            eEDMEW = eEDMEWs[i1E]
            for i1I in xrange(len(eEDMIWs)):
                eEDMISigma = eEDMIWs[i1I]
                for i2E in xrange(len(iEDMEWs)):
                    iEDMEW = iEDMEWs[i2E]
                    for i2I in xrange(len(iEDMIWs)):
                        iEDMISigma = iEDMIWs[i2I]
                        params.append((eEDMEW, eEDMISigma, iEDMEW,
                                                   iEDMISigma))
        return(params)

    ifEDIntegrator1 = \
     IFEnsembleDensityIntegratorRNFWI(nVSteps=nVSteps, leakage=leakage, hMu=hMu,
                                                       hSigma=hSigma, 
                                                       kappaMu=kappaMu, 
                                                       kappaSigma=kappaSigma)
    ifEDIntegrator2 = \
     IFEnsembleDensityIntegratorRNFNI(nVSteps=nVSteps, leakage=leakage, hMu=hMu, 
                                                       hSigma=hSigma)

    def sinusoidalInputMeanFrequency(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    def eEDMISigma(t, w):
        r = ifEDIntegrator2.getSpikeRate(t=t-dt)
        return(w*r)

    def iEDMESigma(t, w):
        r = ifEDIntegrator1.getSpikeRate(t=t-dt)
        return(w*r)

    twoPIFEDIntegrator = \
     TwoPopulationsIFEnsembleDensityIntegrator(ifEDIntegrator1=ifEDIntegrator1,
                                                ifEDIntegrator2=ifEDIntegrator2)
    spikeRatesForParamsCalculator = \
     SpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs(t0=t0, 
                                                          tf=tf, 
                                                          dt=dt,
                                                          nVSteps=nVSteps,
                                                          edm1Rho0=eEDMRho0,
                                                          edm2Rho0=iEDMRho0,
                                                          edm1ESigma=sinusoidalInputMeanFrequency,
                                                          edm1ISigma=eEDMISigma,
                                                          edm2ESigma=iEDMESigma,
                                                          edm2ISigma=None,
                                                          twoPIFEDIntegrator=
                                                           twoPIFEDIntegrator)
    params = generateParams(eEDMEWs=eEDMEWs, eEDMIWs=eEDMIWs, iEDMEWs=iEDMEWs,
                                             iEDMIWs=iEDMIWs)
    times, spikeRates = \
     spikeRatesForParamsCalculator.calculateSpikeRatesForParams(params=params)
    np.savez(resultsFilename, params=params, times=times, spikeRates=spikeRates)

if __name__ == "__main__":
    main(sys.argv)

