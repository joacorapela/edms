
import sys
import numpy as np
import math
import pdb
from IFEnsembleDensityIntegratorRWFWI import IFEnsembleDensityIntegratorRWFWI 
from TwoPopulationsIFEnsembleDensityIntegrator import TwoPopulationsIFEnsembleDensityIntegrator

def main(argv):
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

    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = np.linspace(1, nVSteps, nVSteps)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    eEDMRho0 = rho0
    iEDMRho0 = rho0

    t0 = 0.0
    tf = 1.2
    dt = 1e-3
    dtSaveRhos = 1e-3
    resultsFilename = 'results/twoPSinusoidalRWFWIPopulation.npz'

    ifEDIntegrator1 = \
     IFEnsembleDensityIntegratorRWFWI(nVSteps=nVSteps, 
                                       leakage=leakage, 
                                       hMu=hMu, hSigma=hSigma, 
                                       kappaMu=kappaMu, 
                                       kappaSigma=kappaSigma, 
                                       fracExcitatoryNeurons=eFracExcitatoryNeurons,
                                       nInputsPerNeuron=eNInputsPerNeuron)
    ifEDIntegrator2 = \
     IFEnsembleDensityIntegratorRWFWI(nVSteps=nVSteps,
                                       leakage=leakage, 
                                       hMu=hMu, hSigma=hSigma, 
                                       kappaMu=kappaMu, 
                                       kappaSigma=kappaSigma, 
                                       fracExcitatoryNeurons=iFracExcitatoryNeurons,
                                       nInputsPerNeuron=iNInputsPerNeuron)

    def sinusoidalInputMeanFrequency(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    def eEDMESigma(t):
        return(sinusoidalInputMeanFrequency(t))

    def eEDMISigma(t, w=15):
        r = ifEDIntegrator2.getSpikeRate(t=t-dt)
        return(w*r)

    def iEDMESigma(t, w=50):
        r = ifEDIntegrator1.getSpikeRate(t=t-dt)
        return(w*r)

    twoPIFEDIntegrator = \
     TwoPopulationsIFEnsembleDensityIntegrator(ifEDIntegrator1=ifEDIntegrator1,
                                                ifEDIntegrator2=ifEDIntegrator2)
    twoPIFEDIntegrator.prepareToIntegrate(t0=t0,
                                           tf=tf,
                                           dt=dt,
                                           nVSteps=nVSteps,
                                           edm1EInputCurrent=eEDMESigma,
                                           edm1IInputCurrent=eEDMISigma,
                                           edm2EInputCurrent=iEDMESigma,
                                           edm2IInputCurrent=None)
    twoPIFEDIntegrator.setInitialValue(edm1Rho0=eEDMRho0, edm2Rho0=iEDMRho0)
    eEDMTimes, eEDMRhos, eEDMSpikeRates, iEDMTimes, iEDMRhos, iEDMSpikeRates, \
     saveRhosTimeDSFactor = twoPIFEDIntegrator.integrate(dtSaveRhos=dtSaveRhos)

    np.savez(resultsFilename, vs=vs, 
                              eTimes=eEDMTimes, 
                              eRhos=eEDMRhos,
                              eSpikeRates=eEDMSpikeRates, 
                              iTimes=iEDMTimes, 
                              iRhos=iEDMRhos,
                              iSpikeRates=iEDMSpikeRates, 
                              saveRhosTimeDSFactor=saveRhosTimeDSFactor,
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

