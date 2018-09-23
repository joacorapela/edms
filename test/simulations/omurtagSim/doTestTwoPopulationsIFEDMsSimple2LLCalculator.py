
import sys
import numpy as np
import math
import pdb
from ifEDMsFunctions import computeA0, computeA1, computeA2, computeQRs
from TwoPopulationsIFEDMsSimpleGradientCalculator import TwoPopulationsIFEDMsSimpleGradientCalculator
from TwoPopulationsIFEnsembleDensitySimpleIntegrator import TwoPopulationsIFEnsembleDensitySimpleIntegrator
from LLCalculator import LLCalculator

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
    eRho0 = rho0
    iRho0 = rho0

    t0 = 0.0
    tfSpikeRatesForParams = 0.25
    tfSaveYs = 0.3
    tf = min(tfSpikeRatesForParams, tfSaveYs)
    dt = 1e-5

    wEIsStart = 10
    wEIsStop = 100
    wEIsStep = 10
    wEIs = np.arange(start=wEIsStart, stop=wEIsStop, step=wEIsStep)
    wIEsStart = 5
    wIEsStop = 50
    wIEsStep = 5
    wIEs = np.arange(start=wIEsStart, stop=wIEsStop, step=wIEsStep)

    trueWEI = 50.0
    trueWIE = 15.0
    ysSigma = 2.0
#     ysSigma = 1e-6
    spikesRatesForParamsFilename = \
     "results/spikeRatesForParamsOfTwoPopulationsSimple2T0%.02fTf%.02fDt%fWEIs%.02f-%.02f-%.02fWIEs%.02f-%.02f-%.02f.npz" % (t0, tfSpikeRatesForParams, dt, wEIsStart, wEIsStop, wEIsStep, wIEsStart, wIEsStop, wIEsStep)
    ysFilename = "results/simple2YsForTwoPopulationsT0%.02fTf%.02fDt%fWEI%.02fWIE%.02fSigma%.02f.npz" % (t0, tfSaveYs, dt, trueWEI, trueWIE, ysSigma)
    resultsFilename = "results/twoPopulationsSimple2LLsT0%.02fTf%.02fDt%fTrueWEI%.02fTrueWIE%.02fWEI%.02f-%.02f-%.02fWIE%.02f-%.02f-%.02fSigma%.02f.npz" % (t0, tf, dt, trueWEI, trueWIE, wEIsStart, wEIsStop, wEIsStep, wIEsStart, wIEsStop, wIEsStep, ysSigma)

    loadRes = np.load(spikesRatesForParamsFilename)
    params = loadRes["params"]
    spikeRatesForParams = loadRes["spikeRates"]
    
    loadRes = np.load(ysFilename)
    ys = loadRes["ys"]
    sigma = loadRes["sigma"]
    trueParams = loadRes["params"]

    nSamples = int(round(tf/dt))
    spikeRatesForParams = spikeRatesForParams[:, :nSamples, :]
    ys = ys[:nSamples,]
    llCalculator = LLCalculator()
    lls = llCalculator.calculateLLsForParams(ys=ys, 
                                             spikeRatesForParams=
                                              spikeRatesForParams, sigma=sigma)
    np.savez(resultsFilename, params=params, lls=lls)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

