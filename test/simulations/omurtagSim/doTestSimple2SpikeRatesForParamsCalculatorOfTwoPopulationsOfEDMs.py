
import sys
import numpy as np
import math
import pdb
from ifEDMsFunctions import computeA0, computeA1, computeA2, computeQRs
from TwoPopulationsIFEnsembleDensitySimple2Integrator import TwoPopulationsIFEnsembleDensitySimple2Integrator
from SimpleSpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs import \
 SimpleSpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    nInputsPerNeuron = 10
    fracExcitatoryNeurons = 0.2
    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    eRho0 = rho0
    iRho0 = rho0
    t0 = 0.0
#     tf = 1.0
    tf = 0.25
    dt = 1e-5
    wIEsStart = 5
    wIEsStop = 50
    wIEsStep = 5
    wEIsStart = 10
    wEIsStop = 100
    wEIsStep = 10
    wIEs = np.arange(start=wIEsStart, stop=wIEsStop, step=wIEsStep)
    wEIs = np.arange(start=wEIsStart, stop=wEIsStop, step=wEIsStep)
    wIIs = [None]

    resultsFilename =
"results/spikeRatesForParamsOfTwoPopulationsSimple2T0%.02fTf%.02fDt%fWEIs%.02d-%.02f-%.02fWIEs%.02fd-%.02fd-%.02fd.npz" % (t0, tf, dt, wEIsStart, wEIsStop, wEIsStep, wIEsStart, wIEsStop, wIEsStep)

    def generateParams(wEIs, wIEs):
        params = []

        for wEI in wEIs:
            for wIE in wIEs:
                params.append((wEI, wIE))
        return(params)

    def eInput(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    times = np.arange(start=t0, stop=tf+dt, step=dt)
    eInputs = eInput(t=times)

    a0 = computeA0(nVSteps=nVSteps, leakage=leakage)
    a1 = computeA1(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu, kappaSigma=kappaSigma)
    reversedQs = computeQRs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)

    twoPIFEDIntegrator = TwoPopulationsIFEnsembleDensitySimple2Integrator(\
                          a0=a0, a1=a1, a2=a2, dt=dt, dv=dv, 
                          reversedQs=reversedQs, 
                          nInputsPerNeuron=nInputsPerNeuron, 
                          fracExcitatoryNeurons=fracExcitatoryNeurons)
    spikeRatesForParamsCalculator = \
     SimpleSpikeRatesForParamsCalculatorOfTwoPopulationsOfEDMs(\
      eInputs=eInputs, eRho0=eRho0, iRho0=iRho0, 
      twoPIFEDIntegrator=twoPIFEDIntegrator)
    params = generateParams(wEIs=wEIs, wIEs=wIEs)
    spikeRates = \
     spikeRatesForParamsCalculator.calculateSpikeRatesForParams(params=params)
    np.savez(resultsFilename, params=params, times=times, spikeRates=spikeRates)

if __name__ == "__main__":
    main(sys.argv)

