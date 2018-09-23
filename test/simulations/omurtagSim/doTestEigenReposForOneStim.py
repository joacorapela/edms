
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


    sigma0 = 800
    b = 0.6
#     minStim = sigma0*(1-b)
#     maxStim = sigma0*(1+b)
    minStim = 100.0
    maxStim = 109.0
#     maxStim = 424
#     nStimSteps = 100
#     stepStim = (maxStim-minStim)/nStimSteps
    stepStim = 4.0
#     stepStim = 50.0
    nEigen = 17

    ifEDIntegrator = IFEnsembleDensityIntegratorRNFNI(leakage=leakage, 
                                                       hMu=hMu,
                                                       hSigma=hSigma,
                                                       nVSteps=nVSteps)
    a0 = ifEDIntegrator._a0
    a1 = ifEDIntegrator._a1
    eigenRepos = EigenReposForOneStim()
    eigenRepos.computeEigenDecompositions(a0=a0, a1=a1, minStim=minStim,
                                                 maxStim=maxStim,
                                                 stepStim=stepStim,
                                                 nEigen=nEigen)

    eValsAtMinStim = eigenRepos.getEigenvalues(s=minStim)
    eValsAtMinStimPlusStep = eigenRepos.getEigenvalues(s=minStim+
                                                          stepStim)
    eValsAtMinStimPlusHalfStep = \
     eigenRepos.getEigenvalues(s=minStim+0.5*stepStim)

    plt.plot(eValsAtMinStim, label=r'$\lambda$s @ min')
    plt.plot(eValsAtMinStimPlusHalfStep, label=r'$\lambda$s @ min+step/2')
    plt.plot(eValsAtMinStimPlusStep, label=r'$\lambda$s @ min+step')
    plt.grid()
    plt.legend()

    eVecsAtMinStim = eigenRepos.getEigenvectors(s=minStim)
    eVecsAtMinStimPlusStep = eigenRepos.getEigenvectors(s=minStim+
                                                          stepStim)
    eVecsAtMinStimPlusHalfStep = \
     eigenRepos.getEigenvectors(s=minStim+0.5*stepStim)

#     plt.figure()
#     eVecIndex = 1
#     plt.clf()
#     plt.plot(vs, eVecsAtMinStim[:, eVecIndex], label=r'$\phi_{%d}$ @ min' % eVecIndex)
#     plt.plot(vs, eVecsAtMinStimPlusHalfStep[:, eVecIndex], label=r'$\phi_{%d}$ @ min+step/2' % eVecIndex)
#     plt.plot(vs, eVecsAtMinStimPlusStep[:, eVecIndex], label=r'$\phi_{%d}$ @ min+step' % eVecIndex)
#     plt.grid()
#     plt.legend()

    plt.figure()
    eVecIndicesToPlot = [0, 1, 2, 3, 4, 5]
    for i in xrange(len(eVecIndicesToPlot)):
        eVecIndex = eVecIndicesToPlot[i]
        plt.plot(vs, eVecsAtMinStimPlusStep[:, eVecIndex],
                     label=r'$\phi_{%d}$' % eVecIndex)
    plt.grid()
    plt.legend()

    plt.show()

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

