
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
from myUtils import plotSpikeRates, plotRhos
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI
from LargeEigenReposForOneStim import LargeEigenReposForOneStim
from ifEDMsFunctions import computeQs

def main(argv):
    eigenReposFilename = 'results/aLargeEigenResposForOneStimRNFNI.pickle'
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
#     n1 = 6.0
#     kappaMu = 0.03
#     kappaSigma = kappaMu*0.3
#     nInputsPerNeuron = 50
#     fracExcitatoryNeurons = 0.2
#     rho0 = np.zeros(nVSteps)
#     rho0[0] = 1.0*nVSteps
    dv = 1.0/nVSteps
    mu = 0.25
    sigma2 = (0.01)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    spikeRate0 = 0.0
    t0 = 0.0
#     tf = 1.0
    tf = 0.20
    dt = 1e-5
    startTimePlotRhos = 0.05
#     resultsFilename = '/tmp/sinusoidalRWFWIPopulationKMu%.4f.npz'% kappaMu
#     spikeRatesFigsFilename = '/tmp/spikeRatesSinusoidalRWFWIPopulationKMu%.4f.eps' % kappaMu
#     rhosFigsFilename = '/tmp/rhosSinusoidalRWFWIPopulationKMu%.4f.eps' % kappaMu
#     resultsFilename = 'results/stepRWFWIPopulation.npz'
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRWFWIPopulation.eps'
#     rhosFigsFilename = 'figures/rhosStepRWFWIPopulation.eps'
    sinusoidalInputFreq = 4


    sigma0 = 800
#     def stepInputMeanFrequency(t, sigma0=sigma0):
#         return(sigma0)
#     minStimStep = sigma0
#     maxStimStep = sigma0+1
#     stepStimStep = 1

    b = 0.6
    freq = 4
    def sinusoidalInputMeanFrequency(t, sigma0=sigma0, b=b, omega=2*np.pi*freq):
        return(sigma0*(1+b*np.sin(omega*t)))
    def dSinusoidalInputMeanFrequency(t, sigma0=sigma0, b=b, 
                                         omega=2*np.pi*freq):
        return(sigma0*b*omega*np.cos(omega*t))

    minStimSinusoidal = sigma0*(1-b)
    maxStimSinusoidal = sigma0*(1+b)
#     maxStimSinusoidal = 325
#     nStimStepsSinusoidal = 100
#     stepStimSinusoidal = (maxStimSinusoidal-minStimSinusoidal)/nStimStepsSinusoidal
    stepStimSinusoidal = 4.0

    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRNFNI(leakage, hMu, hSigma)
    ifEDIntegrator.prepareToIntegrate(rho0=rho0, spikeRate0=spikeRate0,
                                                 t0=t0, tf=tf, dt=dt, 
                                                 nVSteps=nVSteps, 
                                                 eInputCurrent=
                                                  sinusoidalInputMeanFrequency)
#                                                   stepInputMeanFrequency)
    a0 = ifEDIntegrator._a0
    a1 = ifEDIntegrator._a1
    qs = computeQs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    eigenRepos = LargeEigenReposForOneStim(a0=a0, a1=a1, 
                                             minStim=minStimSinusoidal,
                                             maxStim=maxStimSinusoidal,
                                             stepStim=stepStimSinusoidal,
                                             qs=qs)
    with open(eigenReposFilename, 'wb') as f:
        pickle.dump(eigenRepos, f)

if __name__ == "__main__":
    main(sys.argv)

