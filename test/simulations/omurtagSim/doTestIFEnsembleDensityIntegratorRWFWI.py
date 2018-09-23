

import sys
sys.path.append('../../../src/')
sys.path.append('../../../src/omurtagEtAl00')

import numpy as np
import pdb
import matplotlib.pyplot as plt
from plotEDMsResults import plotSpikeRatesAndInputs, plotRhos
from IFEnsembleDensityIntegratorRWFWI import IFEnsembleDensityIntegratorRWFWI

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
    fracExcitatoryNeurons = 0.2
    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    t0 = 0.0
#     tf = 1.20
    tf = 0.25
    dt = 1e-5
    dtSaveRhos = 1e-3
    startTimePlotRhos = 0.05
    resultsFilename = 'results/sinusoidalRWFWIPopulationKMu%.4f.npz'% kappaMu
    spikeRatesFigsFilename = 'figures/spikeRatesSinusoidalRWFWIPopulationKMu%.4f.eps' % kappaMu
    rhosFigsFilename = 'figures/rhosSinusoidalRWFWIPopulationKMu%.4f.eps' % kappaMu
#     resultsFilename = 'results/stepRWFWIPopulation.npz'
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRWFWIPopulation.eps'
#     rhosFigsFilename = 'figures/rhosStepRWFWIPopulation.eps'

    def eExternalInput(t, sigma0=800, b=0.5, omega=8*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    def iExternalInput(t, sigma0=200, b=0.5, omega=8*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRWFWI(nVSteps=nVSteps, leakage=leakage, 
                                                       hMu=hMu, 
                                                       hSigma=hSigma, 
                                                       kappaMu=kappaMu,
                                                       kappaSigma=kappaSigma,
                                                       fracExcitatoryNeurons=
                                                        fracExcitatoryNeurons,
                                                       nInputsPerNeuron=
                                                        nInputsPerNeuron)
    ifEDIntegrator.prepareToIntegrate(t0=t0, tf=tf, dt=dt, 
                                             eExternalInput=eExternalInput,
                                             iExternalInput=iExternalInput)
    ifEDIntegrator.setInitialValue(rho0=rho0)
    ts, rhos, spikeRates, saveRhosTimeDSFactor, \
    eExternalInputsArray, eFeedbackInputsArray, \
    iExternalInputsArray, iFeedbackInputsArray = \
     ifEDIntegrator.integrate(dtSaveRhos=dtSaveRhos)

    np.savez(resultsFilename, spikeRates=spikeRates, ts=ts, vs=vs,
                              rhos=rhos, 
                              saveRhosTimeDSFactor=saveRhosTimeDSFactor,
                              eExternalInputHist=eExternalInputsArray,
                              eFeedbackInputHist=eFeedbackInputsArray,
                              iExternalInputHist=iExternalInputsArray,
                              iFeedbackInputHist=iFeedbackInputsArray)
    averageWinTimeLength = 1e-3                  
    plt.figure()
    plotSpikeRatesAndInputs(times=ts, spikeRates=spikeRates, 
                                      eExternalInput=eExternalInputsArray,
                                      eFeedbackInput=eFeedbackInputsArray,
                                      iExternalInput=iExternalInputsArray,
                                      iFeedbackInput=iFeedbackInputsArray,
                                      dt=dt, 
                                      averageWinTimeLength=averageWinTimeLength)
    plt.savefig(spikeRatesFigsFilename)
    startSamplePlotRhos = startTimePlotRhos/dtSaveRhos
    plt.figure()
    plotRhos(vs, ts[::int(saveRhosTimeDSFactor)][int(startSamplePlotRhos):], rhos[:, int(startSamplePlotRhos):])
    plt.savefig(rhosFigsFilename)

    pdb.set_trace()
    
if __name__ == "__main__":
    main(sys.argv)

