
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
from myUtils import plotSpikeRates, plotRhos
from LDIFEnsembleDensityIntegratorRWFWI \
 import LDIFEnsembleDensityIntegratorRWFWI

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

    t0 = 0.0
    tf = 1.20
    dt = 1e-5
    dtSaveLDCoefs = 1e-3
    eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom0.000000EStimTo4000.000000EStimStep20.000000IStimFrom0.000000IStimTo600.000000IStimStep20.000000NEigen17.pickle"
    nEigen = 17
    resultsFilename = "results/sinusoidalLDRWFWIPopulationNEigen%02d.npz" % (nEigen)
    spikeRatesFigsFilename = "figures/spikeRatesSinusoidalLDRWFWIPopulationNEigen%02d.eps" % (nEigen)

    def sinusoidalInputMeanFrequency(t, sigma0=800, b=0.5, omega=6*np.pi): 
        return(sigma0*(1+b*np.sin(omega*t)))

    with open(eigenReposFilename, "rb") as f:
        eigenRepos = pickle.load(f)
    ldIFEDIntegrator = \
     LDIFEnsembleDensityIntegratorRWFWI(nVSteps=nVSteps, leakage=leakage,
                                                         hMu=hMu,
                                                         hSigma=hSigma, 
                                                         kappaMu=kappaMu,
                                                         kappaSigma=kappaSigma,
                                                         fracExcitatoryNeurons=
                                                          fracExcitatoryNeurons,
                                                         nInputsPerNeuron=
                                                          nInputsPerNeuron,
                                                         nEigen=nEigen,
                                                         eigenRepos=eigenRepos)
    ldIFEDIntegrator.prepareToIntegrate(t0=t0, tf=tf, dt=dt,
                                               eInputCurrent=sinusoidalInputMeanFrequency,
                                               iInputCurrent=None)
    ldIFEDIntegrator.setInitialValue(rho0=rho0)
    ts, sriLDCoefsCol, spikeRates, saveLDCoefsTimeDSFactor = \
     ldIFEDIntegrator.integrate(dtSaveLDCoefs=dtSaveLDCoefs)
    np.savez(resultsFilename, spikeRates=spikeRates, ts=ts, vs=vs, 
                              sriLDCoefsCol=sriLDCoefsCol,
                              saveLDCoefsTimeDSFactor=saveLDCoefsTimeDSFactor,
                              eInputCurrentHist=\
                               ldIFEDIntegrator.getEInputCurrentHist(),
                              eFeedbackCurrentHist=\
                               ldIFEDIntegrator.getEFeedbackCurrentHist(),
                              iInputCurrentHist=\
                                ldIFEDIntegrator.getIInputCurrentHist(),
                               iFeedbackCurrentHist=\
                                ldIFEDIntegrator.getIFeedbackCurrentHist())

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

