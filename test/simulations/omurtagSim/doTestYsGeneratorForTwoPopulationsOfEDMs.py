
import sys
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
from IFEnsembleDensityIntegratorRNFWI import IFEnsembleDensityIntegratorRNFWI 
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI 
from TwoPopulationsIFEnsembleDensityIntegrator import TwoPopulationsIFEnsembleDensityIntegrator
from SpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs import \
 SpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs
import pickle

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
    tf = 0.40
    dt = 1e-5
    wEE = None
    wIE = 15
    wEI = 50
    wII = None
    sigma = 0.01
    resultsFilename = "results/ysForTwoPopulationsT0%.02fTf%.02fDt%fWEE%sWIE%sWEI%sWII%sSigma%.02f.npz" % (t0, tf, dt, str(wEE), str(wIE), str(wEI), str(wII), sigma)

    def addUncorrelatedGaussianNoiseToSpikesRatesForParams(srsForPops, sigma):
        nsrsForPops = np.empty(srsForPops.shape)
        for j in xrange(srsForPops.shape[1]):
            srsForPop = srsForPops[:, j]
            nsrsForPops[:, j] = srsForPop + np.random.normal(loc=0, 
                                                              scale=sigma,
                                                              size=
                                                               srsForPop.size)
        return(nsrsForPops)


    ifEDIntegrator1 = \
     IFEnsembleDensityIntegratorRNFWI(nVSteps=nVSteps, leakage=leakage, hMu=hMu, 
                                                       hSigma=hSigma, kappaMu=kappaMu, 
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
     SpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs(\
      t0=t0, 
      tf=tf, 
      dt=dt,
      nVSteps=nVSteps,
      edm1Rho0=eEDMRho0,
      edm2Rho0=iEDMRho0,
      edm1ESigma=sinusoidalInputMeanFrequency,
      edm1ISigma=eEDMISigma,
      edm2ESigma=iEDMESigma,
      edm2ISigma=None,
      twoPIFEDIntegrator=twoPIFEDIntegrator)
    params = [(wEE, wIE, wEI, wII)]
    times, srsForParams = spikeRatesForParamsCalculator.\
     calculateSpikeRatesForParams(params=params)
    with open("spikeRatesForParams.pickle", 'wb') as f:
        pickle.dump((times, srsForParams), f)
#     with open("spikeRatesForParams.pickle", 'rb') as f:
#         (times, srsForParams) = pickle.load(f)
    srsForPops = np.reshape(srsForParams[0, :, :], srsForParams.shape[1:])
    ys = addUncorrelatedGaussianNoiseToSpikesRatesForParams(srsForPops=
                                                              srsForPops, 
                                                             sigma=sigma)
    np.savez(resultsFilename, sigma=sigma, params=params, ys=ys, color="blue")
    plt.plot(times, ys[:, 0], label="noisy")
    plt.plot(times, srsForPops[:, 0], label="noiseless", color="red")
    plt.legend()
    plt.show()
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

