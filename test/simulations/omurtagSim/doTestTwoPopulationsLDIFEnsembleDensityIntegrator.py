
import sys
import numpy as np
import math
import pickle
import pdb
from LDIFEnsembleDensityIntegratorRWFWI import \
 LDIFEnsembleDensityIntegratorRWFWI 
from LDIFEnsembleDensityIntegratorRWFNI import \
 LDIFEnsembleDensityIntegratorRWFNI 
from TwoPopulationsLDIFEnsembleDensityIntegrator import \
 TwoPopulationsLDIFEnsembleDensityIntegrator

def main(argv):
    if(len(argv)>1):
        nEigen = int(argv[1])
    else:
        nEigen = 17
        print("Usage: %s <number of moving basis>" % argv[0])
        print("Using default of %d number of basis" % nEigen)

    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    eNInputsPerNeuron = 10
    iNInputsPerNeuron = 10
    eFracExcitatoryNeurons = 0.2
    iFracExcitatoryNeurons = 0.2

    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
#     rho0 = np.zeros(nVSteps)
#     rho0[0] = 1.0*nVSteps
    edm1Rho0 = rho0
    edm2Rho0 = rho0

    t0 = 0.0
    tf = 1.2
    dt = 1e-5
    dtSaveLDCoefs = 1e-3
    eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom0.000000EStimTo4000.000000EStimStep20.000000IStimFrom0.000000IStimTo600.000000IStimStep20.000000NEigen17.pickle"
    resultsFilename = 'results/twoPSinusoidalRWFWIPopulationNEigen%02d.npz' % (nEigen)

    with open(eigenReposFilename, "rb") as f:
        eigenRepos = pickle.load(f)
    ifEDIntegrator1 = \
     LDIFEnsembleDensityIntegratorRWFWI(nVSteps=nVSteps,
                                         leakage=leakage, 
                                         hMu=hMu, hSigma=hSigma, 
                                         kappaMu=kappaMu, kappaSigma=kappaSigma,
                                         fracExcitatoryNeurons=\
                                          eFracExcitatoryNeurons,
                                         nInputsPerNeuron=eNInputsPerNeuron,
                                         nEigen=nEigen,
                                         eigenRepos=eigenRepos)
    ifEDIntegrator2 = \
     LDIFEnsembleDensityIntegratorRWFWI(nVSteps=nVSteps,
                                         leakage=leakage, 
                                         hMu=hMu, hSigma=hSigma, 
                                         kappaMu=kappaMu, kappaSigma=kappaSigma,
                                         fracExcitatoryNeurons=\
                                          iFracExcitatoryNeurons,
                                         nInputsPerNeuron=iNInputsPerNeuron,
                                         nEigen=nEigen,
                                         eigenRepos=eigenRepos)
    def sinusoidalInputMeanFrequency(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    def edm1ESigma(t):
        return(sinusoidalInputMeanFrequency(t))

    def edm1ISigma(t, w=15):
        r = ifEDIntegrator2.getSpikeRate(t=t-dt)
        return(w*r)

    def edm2ESigma(t, w=50):
        r = ifEDIntegrator1.getSpikeRate(t=t-dt)
        return(w*r)

    twoPIFEDIntegrator = \
     TwoPopulationsLDIFEnsembleDensityIntegrator(ifEDIntegrator1=ifEDIntegrator1,
                                                ifEDIntegrator2=ifEDIntegrator2)
    twoPIFEDIntegrator.prepareToIntegrate(t0=t0,
                                           tf=tf,
                                           dt=dt,
                                           nVSteps=nVSteps,
                                           edm1EInputCurrent=edm1ESigma,
                                           edm2EInputCurrent=edm2ESigma,
                                           edm1IInputCurrent=edm1ISigma,
                                           edm2IInputCurrent=None)
    twoPIFEDIntegrator.setInitialValue(edm1Rho0=edm1Rho0, edm2Rho0=edm2Rho0)
    edm1Times, edm1SRILDCoefs,  edm1SpikeRates, \
    edm2Times, edm2SRILDCoefs,  edm2SpikeRates, saveLDCoefsTimesDSFactor = \
     twoPIFEDIntegrator.integrate(dtSaveLDCoefs=dtSaveLDCoefs)

    np.savez(resultsFilename, vs=vs,
                              eTimes=edm1Times, 
                              eSRILDCoefs=edm1SRILDCoefs,
                              eSpikeRates=edm1SpikeRates, 
                              iTimes=edm2Times, 
                              iSRILDCoefs=edm2SRILDCoefs,
                              iSpikeRates=edm2SpikeRates, 
                              saveLDCoefsTimesDSFactor=saveLDCoefsTimesDSFactor,
                              eEInputCurrentHist=\
                               ifEDIntegrator1.getEInputCurrentHist(), 
                              eEFeedbackCurrentHist=\
                               ifEDIntegrator1.getEFeedbackCurrentHist(), 
                              eIInputCurrentHist=\
                               ifEDIntegrator1.getIInputCurrentHist(), 
                              eIFeedbackCurrentHist=\
                               ifEDIntegrator1.getIFeedbackCurrentHist(), 
                              iEInputCurrentHist=\
                               ifEDIntegrator2.getEInputCurrentHist(), 
                              iEFeedbackCurrentHist=\
                               ifEDIntegrator2.getEFeedbackCurrentHist(), 
                              iIInputCurrentHist=\
                               ifEDIntegrator2.getIInputCurrentHist(), 
                              iIFeedbackCurrentHist=\
                               ifEDIntegrator2.getIFeedbackCurrentHist())


if __name__ == "__main__":
    main(sys.argv)

