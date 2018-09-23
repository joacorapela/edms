
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from myUtils import plotSpikeRates, plotRhos
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI
from EigenReposForOneStim import EigenReposForOneStim
from ifEDMsFunctions import computeQs

def main(argv):
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
    tf = 0.01
    dt = 1e-5
    startTimePlotRhos = 0.05
#     resultsFilename = '/tmp/sinusoidalRWFWIPopulationKMu%.4f.npz'% kappaMu
#     spikeRatesFigsFilename = '/tmp/spikeRatesSinusoidalRWFWIPopulationKMu%.4f.eps' % kappaMu
#     rhosFigsFilename = '/tmp/rhosSinusoidalRWFWIPopulationKMu%.4f.eps' % kappaMu
#     resultsFilename = 'results/stepRWFWIPopulation.npz'
#     spikeRatesFigsFilename = 'figures/spikeRatesStepRWFWIPopulation.eps'
#     rhosFigsFilename = 'figures/rhosStepRWFWIPopulation.eps'
    sinusoidalInputFreq = 4
    nEigen = nVSteps
    nEigenR = 15


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
#     maxStimSinusoidal = sigma0*(1+b)
    maxStimSinusoidal = 325
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
    eigenRepos = EigenReposForOneStim(a0=a0, a1=a1, 
                                             minStim=minStimSinusoidal,
                                             maxStim=maxStimSinusoidal,
                                             stepStim=stepStimSinusoidal,
                                             nEigen=nEigen, qs=qs)
#     ts = np.arange(t0, tf, dt)
#     for i in xrange(len(ts)):
#         t = ts[i]
#         sigmaE = sinusoidalInputMeanFrequency(t=t)
#         reducedEVals, reducedM = \
#          eigenRepos.getReducedEigenvaluesAndMMatrix(s=sigmaE)
#         dSigmaE = dSinusoidalInputMeanFrequency(t=t)
#         diffMatrix = np.diag(reducedEVals) + dSigmaE*reducedM
#         eigRes = np.linalg.eig(diffMatrix)
#         maxRealEigVals = max(eigRes[0].real)
#         print("maxRealEigVal=%f" % maxRealEigVals)
#         pdb.set_trace()
#     pdb.set_trace()

    diffMatrix = a0+sinusoidalInputMeanFrequency(t=0)*a1
    eigenvalues, eigenvectors, aEigenvectors = \
     eigenRepos.getEigenResults(diffMatrix)
    ldCoefs = aEigenvectors.transpose().conjugate().dot(rho0)
    rRho0 = eigenvectors[:, :nEigenR].dot(ldCoefs[:nEigenR])

#     plt.plot(eigenvalues)
#     plt.figure()
    plt.plot(vs, rho0, label=r'$\rho_0$')
    plt.plot(vs, rRho0, label=r'$\hat\rho_0$')
    plt.grid()
    plt.legend()
    plt.show()

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

