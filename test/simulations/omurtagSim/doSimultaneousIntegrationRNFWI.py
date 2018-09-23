# This version works for two stimulus and real integration

import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
from myUtils import plotSpikeRates, plotRhos, \
     splitRealAndImaginaryPartsInVector, splitRealAndImaginaryPartsInMatrix, \
     getRealPartOfCArrayDotSRIVector
from ifEDMsFunctions import computeQs
from IFEnsembleDensityIntegratorRNFWI import IFEnsembleDensityIntegratorRNFWI
from EigenReposForOneStim import EigenReposForOneStim

def main(argv):
    plt.ion()

    if(len(argv)>1):
        nEigen = int(argv[1])
    else:
        nEigen = 17
        print("Usage: %s <number of moving basis>" % argv[0])
        print("Using default of %d number of basis" % nEigen)

    eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom300.000000EStimTo1500.000000EStimStep20.000000IStimFrom80.000000IStimTo320.000000IStimStep20.000000NEigen17.pickle"
#     eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom300.000000EStimTo1500.000000EStimStep20.000000IStimFrom0.000000IStimTo320.000000IStimStep20.000000NEigen17.pickle"
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    dv = 1.0/nVSteps

    mu = 0.25
    sigma2 = (0.01)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    spikeRate0 = 0.0
    t0 = 0.0
    tf = 1.0
#     tf = 0.2
    dt = 1e-5
    startTimePlotRhos = 0.05
    startSamplePlotRhos = startTimePlotRhos/dt
    reversedQs = computeQs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)[-1::-1]

    sigma0E = 800
    freqE = 4
    sigma0I = 200
    freqI = 1

    b = 0.6
    def sinusoidalInputMeanFrequency(t, sigma0, omega, b=b, phase=-np.pi/2):
        return(sigma0*(1+b*np.sin(omega*t+phase)))

    def constantInputMeanFrequency(t, sigma0):
        return(sigma0)

    linearStimSlope = 1
    def linearInputMeanFrequency(t, sigma0, slope=linearStimSlope):
        return(sigma0+t*linearStimSlope)

    eStim = lambda t: sinusoidalInputMeanFrequency(t=t, sigma0=sigma0E,
                                                        omega=2*np.pi*freqE)
    iStim = lambda t: sinusoidalInputMeanFrequency(t=t, sigma0=sigma0I,
                                                        omega=2*np.pi*freqI)

#     stim = constantInputMeanFrequency

#     stim = linearInputMeanFrequency

    minEStim = sigma0E*(1-b)
    maxEStim = sigma0E*(1+b)
    minIStim = sigma0I*(1-b)
    maxIStim = sigma0I*(1+b)
#     maxStim = 325
#     nStimSteps = 100
#     stepStim = (maxStim-minStim)/nStimSteps
    stepStim = 4.0


    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRNFWI(nVSteps=nVSteps, leakage=leakage, 
                                                       hMu=hMu, hSigma=hSigma, 
                                                       kappaMu=kappaMu,
                                                       kappaSigma=kappaSigma)
    a0 = ifEDIntegrator._a0
    a1 = ifEDIntegrator._a1
    a2 = ifEDIntegrator._a2
    with open(eigenReposFilename, 'rb') as f:
        eigenRepos = pickle.load(f)

    ts = np.arange(t0, tf, dt)
    rhosCol = np.empty([rho0.size, ts.size])
    rhosFromDECol = np.empty([rho0.size, ts.size])
    coefsColFromDE = np.empty([2*nEigen, ts.size])
    sigmaEs = np.empty(ts.size)
    sigmaEs[:] = np.nan
    sigmaIs = np.empty(ts.size)
    sigmaIs[:] = np.nan
    spikeRatesFromRhos = np.empty(ts.size)
    spikeRatesFromDE = np.empty(ts.size)
    spikeRatesFromRhos[:] = np.nan
    spikeRatesFromDE[:] = np.nan

    i = 0
    t = ts[i]
    sigmaE = eStim(t=t)
    sigmaI = iStim(t=t)

    rho = rho0
    rhosCol[:, 0] = rho
    prevRho = rho

    eVals = eigenRepos.getEigenvalues(sE=sigmaE,sI=sigmaI)
    eVecs = eigenRepos.getEigenvectors(sE=sigmaE,sI=sigmaI)
    aEVecs = np.linalg.pinv(eVecs).transpose().conjugate()
    rEVals = eVals[:nEigen]
    rEVecs = eVecs[:, :nEigen]
    rAEVecs = aEVecs[:, :nEigen]
    coefsInBOB = rAEVecs.transpose().conjugate().dot(rho0)
    coefsFromDE = splitRealAndImaginaryPartsInVector(v=coefsInBOB)
    # To avoid loosing probability, lets enforce 
    # coefsFromDE[0]=sum(rho0)/surm(\phi_0) (see Note 12 in knigth00)
    coefsFromDE[0] = sum(rho0)/sum(rEVecs[:, 0].real)
    coefsColFromDE[:, 0] = coefsFromDE
    prevCoefsFromDE = coefsFromDE

    rhoFromDE = getRealPartOfCArrayDotSRIVector(cArray=rEVecs, 
                                                 sriVector=coefsFromDE)
    rhosFromDECol[:, i] = rhoFromDE

    sigmaEs[0] = sigmaE
    sigmaIs[0] = sigmaI
    spikeRatesFromRhos[0] = sigmaE*dv*reversedQs.dot(rho0)
    spikeRatesFromDE[0] = sigmaE*dv*reversedQs.dot(rhoFromDE)

    f, axarr = plt.subplots(3)
    axarr[0].set_title("Time %.04f sec" % ts[0])
    axarr[0].plot(vs, rho0, label="Full", color="blue")
    axarr[0].plot(vs, rhoFromDE, label="%d dims" % nEigen, color="red")
    axarr[0].set_xlabel("Normalized Voltage")
    axarr[0].set_ylabel("Density Value")
    axarr[0].grid()
    axarr[0].legend(loc=2)
    axarr[1].plot(ts, spikeRatesFromRhos, label="Full", color="blue")
    axarr[1].plot(ts, spikeRatesFromDE, label="%d dims" % nEigen, color="red")
    axarr[1].set_xlabel("Time (sec)")
    axarr[1].set_ylabel("Firing Rate (ips)")
    axarr[1].set_xlim(left=t0, right=tf)
    axarr[1].set_ylim(bottom=0, top=50)
    axarr[1].grid()
    axarr[1].legend(loc=9)
    ax1 = axarr[1].twinx()
    ax1.plot(ts, sigmaEs, color="cyan", linestyle="dashed", 
                 label="Input Current")
    ax1.set_ylabel("Firing Rate (ips)")
    ax1.set_ylim(bottom=300, top=1300)
    ax1.legend(loc=1)
    axarr[2].plot(coefsFromDE[::2], color="red")
    axarr[2].set_xlabel("Low-Dimensional Coefficient Index")
    axarr[2].set_ylabel("Real Part of Coefficient")
    axarr[2].grid()

    plt.draw()
#     pdb.set_trace()

    identityHD = np.identity(a0.shape[0])
    identityLD = np.identity(nEigen)
    for i in xrange(1, len(ts)):
        t = ts[i]
        if i%100==0:
            print("Processing step %d (out of %d)" % (i, len(ts)))
        sigmaE = eStim(t=t)
        sigmaI = iStim(t=t)

        rhosDiffMatrix = a0+sigmaE*a1-sigmaI*a2
        rho = (identityHD+dt*rhosDiffMatrix).dot(prevRho)
        rhosCol[:, i] = rho
        prevRho = rho

        eVals = eigenRepos.getEigenvalues(sE=sigmaE, sI=sigmaI)
        eVecs = eigenRepos.getEigenvectors(sE=sigmaE, sI=sigmaI)

        rPrevEVals = rEVals
        rPrevEVecs = rEVecs
        rEVals = eVals[:nEigen]
        rEVecs = eVecs[:, :nEigen]

        diagPrevREVals = np.diag(1+dt*rPrevEVals)
        diffMatrixForDE = np.linalg.pinv(rEVecs).dot(rPrevEVecs).\
                           dot(diagPrevREVals)
        sriDiffMatrixForDE = \
         splitRealAndImaginaryPartsInMatrix(m=diffMatrixForDE)
        coefsFromDE =  sriDiffMatrixForDE.dot(prevCoefsFromDE)
        # To avoid loosing probability, lets enforce 
        # coefsFromDE[0]=sum(rho0)/surm(\phi_0) (see Note 12 in knigth00)
        coefsFromDE[0] = sum(rho0)/sum(rEVecs[:, 0].real)
        coefsColFromDE[:, i] = coefsFromDE
        prevCoefsFromDE = coefsFromDE

        rhoFromDE = getRealPartOfCArrayDotSRIVector(cArray=rEVecs, 
                                                     sriVector=coefsFromDE)
        rhosFromDECol[:, i] = rhoFromDE

        sigmaEs[i] = sigmaE
        sigmaIs[i] = sigmaI
        spikeRatesFromRhos[i] = sigmaE*dv*reversedQs.dot(rho)
        spikeRatesFromDE[i] = sigmaE*dv*reversedQs.dot(rhoFromDE)
        if i%100==0:
            axarr[0].cla()
            axarr[0].set_title("Time %.04f sec" % ts[i])
            axarr[0].plot(vs, rho, label="Full", color="blue")
            axarr[0].plot(vs, rhoFromDE, label="%d dims" % nEigen,
                              color="red")
            axarr[0].grid()
            axarr[0].legend(loc=2)
            axarr[0].set_xlabel("Normalized Voltage")
            axarr[0].set_ylabel(r"$\rho$")
            axarr[1].cla()
            ax1.cla()
            axarr[1].plot(ts, spikeRatesFromRhos, label="Full", color="blue")
            axarr[1].plot(ts, spikeRatesFromDE, label="%d dims" % nEigen,
                              color="red")
            axarr[1].legend(loc=9)
            axarr[1].grid()
            axarr[1].set_xlabel("Time (sec)")
            axarr[1].set_ylabel("Firing Rate (ips)")
            axarr[1].set_xlim(left=t0, right=tf)
            axarr[1].set_ylim(bottom=0, top=50)
            ax1.plot(ts, sigmaEs, color="blue", linestyle='dashed', 
                         label="Exc. Input")
            ax1.plot(ts, sigmaIs, color="cyan", linestyle='dashed', 
                         label="Inh. Input")
            ax1.set_ylabel("Input Current (ips)")
            ax1.set_ylim(bottom=0, top=1300)
            ax1.legend(loc=1)

            axarr[2].cla()
            axarr[2].plot(coefsFromDE[::2], color="red")
            axarr[2].set_xlabel("Low-Dimensional Coefficient Index")
            axarr[2].set_ylabel("Real Part of Coefficient")
            axarr[2].grid()

            plt.draw()
#             pdb.set_trace()

#         pdb.set_trace()
    pdb.set_trace()
#     np.savez(resultsFilename, ts=ts, vs=vs, rhosCol=rhosCol, rhosFromDECol=rhosFromDECol, coefsColFromDE=coefsColFromDE, spikeRatesFromRhos=spikeRatesFromRhos, spikeRatesFromDE=spikeRatesFromDE)

if __name__ == "__main__":
    main(sys.argv)

