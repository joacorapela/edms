
import sys
import numpy as np
import math
import pdb
from SpikeRatesForParamsCalculatorOfOnePopulationOfEDMs import \
 SpikeRatesForParamsCalculatorOfOnePopulationOfEDMs

def main(argv):
    nVSteps = 210
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
#     tf = 1.0
    tf = 0.25
    dt = 1e-5
    leakageStart = 4.0
#     leakageStart = 16.0
    leakageStop = 41.0
#     leakageStop = 21.0
    leakageStep = 4.0
    gStart = 0
#     gStart = 8
    gStop = 19
#     gStop = 11
    gStep = 2
    fStart = 0.0
#     fStart = 0.2
    fStop = 0.45
    fStep = 0.1

    resultsFilename = "results/spikeRatesForParamsOfOnePopulationT0%.02fTf%.02fDt%fLeakage%.02f-%.02f-%.02fNInputsPerNeuron%d-%d-%dFracExcitatoryNeurons%.02f-%.02f-%.02f.npz" % (t0, tf, dt, leakageStart, leakageStop, leakageStep, gStart, gStop, gStep, fStart, fStop, fStep)

    def generateParams(leakageCol, gCol, fCol):
        params = []

        for iLeakage in xrange(len(leakageCol)):
            leakage = leakageCol[iLeakage]
            for iNInputsPerNeuron in xrange(len(gCol)):
                g = gCol[iNInputsPerNeuron]
                for iFracExcitatoryNeuronsCol in xrange(len(fCol)):
                    f = fCol[iFracExcitatoryNeuronsCol]
                    params.append((leakage, g, f))
        return(params)

    def eExternalInput(t, sigma0=800, b=0.5, omega=8*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    def iExternalInput(t, sigma0=200, b=0.5, omega=4*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    spikeRatesForParamsCalculator = \
     SpikeRatesForParamsCalculatorOfOnePopulationOfEDMs(hMu=hMu,
                                                         hSigma=hSigma,
                                                         kappaMu=kappaMu,
                                                         kappaSigma=kappaSigma,
                                                         t0=t0, 
                                                         tf=tf, 
                                                         dt=dt,
                                                         nVSteps=nVSteps,
                                                         rho0=rho0,
                                                         eExternalInput=
                                                          eExternalInput,
                                                         iExternalInput=
                                                          iExternalInput)
    leakageCol = np.arange(start=leakageStart, stop=leakageStop, 
                                               step=leakageStep)
    gCol = np.arange(start=gStart, stop=gStop, step=gStep)
    fCol = np.arange(start=fStart, stop=fStop, step=fStep)
    params = generateParams(leakageCol=leakageCol, gCol=gCol, fCol=fCol)
    times, spikeRates = \
     spikeRatesForParamsCalculator.calculateSpikeRatesForParams(params=params)
    np.savez(resultsFilename, params=params, times=times, spikeRates=spikeRates)

if __name__ == "__main__":
    main(sys.argv)

