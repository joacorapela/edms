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
from IFEnsembleDensityIntegratorRWFWI import IFEnsembleDensityIntegratorRWFWI
from EigenReposForOneStim import EigenReposForOneStim

def main(argv):
    plt.ion()

    if(len(argv)==3):
        nEigen = int(argv[1])
        eVecToPlot = int(argv[2])
        if eVecToPlot>nEigen:
            print("Invalid arguments. nEigen=%d should be larger than eVecToPlot=%d" % (nEigen, eVecToPlot))
            system.exit()
    else:
        nEigen = 17
        eVecToPlot = 0
        print("Usage: %s <number of moving basis> <eigenvector to plot>" % argv[0])
        print("Using default of %d number of basis" % nEigen)
        print("Ploting eigenvector %d" % eVecToPlot)

    eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom0.000000EStimTo4000.000000EStimStep20.000000IStimFrom0.000000IStimTo600.000000IStimStep20.000000NEigen17.pickle"
#     eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom0.000000EStimTo2000.000000EStimStep20.000000IStimFrom0.000000IStimTo300.000000IStimStep20.000000NEigen17.pickle"
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
    tf = 1.0
#     tf = 0.2
    dt = 1e-5
    spikeRate0 = 0.0
    startTimePlotRhos = 0.05
    startSamplePlotRhos = startTimePlotRhos/dt
    reversedQs = computeQs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)[-1::-1]

    scale = 1400
    freq = 4
    baseline = 200
    def rSinusoidalInputMeanFrequency(t, scale, baseline,
                                        omega=2*np.pi*freq, phase=np.pi):
        r = baseline+scale*np.sin(omega*t+phase)
        if r<baseline:
            r = baseline
        return(r)

    def zeroInputMeanFrequency(t):
        return(0.0)

    eStim = lambda t: rSinusoidalInputMeanFrequency(t=t, scale=scale,
                                                         baseline=baseline,
                                                        omega=2*np.pi*freq)
    iStim = zeroInputMeanFrequency

    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRWFWI(nVSteps=nVSteps, leakage=leakage, 
                                                       hMu=hMu, hSigma=hSigma, 
                                                       kappaMu=kappaMu,
                                                       kappaSigma=kappaSigma,
                                                       nInputsPerNeuron=
                                                        nInputsPerNeuron,
                                                       fracExcitatoryNeurons=
                                                        fracExcitatoryNeurons)
    a0 = ifEDIntegrator._a0
    a1 = ifEDIntegrator._a1
    a2 = ifEDIntegrator._a2
    with open(eigenReposFilename, 'rb') as f:
        eigenRepos = pickle.load(f)

    ts = np.arange(t0, tf, dt)
    rhosCol = np.empty([rho0.size, ts.size])
    rhosFromDECol = np.empty([rho0.size, ts.size])
    coefsColFromDE = np.empty([2*nEigen, ts.size])

    sigma0Es = np.empty(ts.size)
    sigma0Es[:] = np.nan
    sigma0Is = np.empty(ts.size)
    sigma0Is[:] = np.nan

    sigmaFEsFromRho = np.empty(ts.size)
    sigmaFEsFromRho[:] = np.nan
    sigmaFIsFromRho = np.empty(ts.size)
    sigmaFIsFromRho[:] = np.nan

    sigmaFEsFromDE = np.empty(ts.size)
    sigmaFEsFromDE[:] = np.nan
    sigmaFIsFromDE = np.empty(ts.size)
    sigmaFIsFromDE[:] = np.nan

    sigmaEsFromRho = np.empty(ts.size)
    sigmaEsFromRho[:] = np.nan
    sigmaIsFromRho = np.empty(ts.size)
    sigmaIsFromRho[:] = np.nan

    sigmaEsFromDE = np.empty(ts.size)
    sigmaEsFromDE[:] = np.nan
    sigmaIsFromDE = np.empty(ts.size)
    sigmaIsFromDE[:] = np.nan

    spikeRatesFromRho = np.empty(ts.size)
    spikeRatesFromDE = np.empty(ts.size)
    spikeRatesFromRho[:] = np.nan
    spikeRatesFromDE[:] = np.nan

    i = 0
    t = ts[i]
    sigma0E = eStim(t=t)
    sigma0I = iStim(t=t)

    rho = rho0
    rhosCol[:, 0] = rho
    prevRho = rho

    eVals = eigenRepos.getEigenvalues(sE=sigma0E,sI=sigma0I)
    eVecs = eigenRepos.getEigenvectors(sE=sigma0E,sI=sigma0I)
    aEVecs = np.linalg.pinv(eVecs).transpose().conjugate()
    rEVals = eVals[:nEigen]
    rEVecs = eVecs[:, :nEigen]
    rAEVecs = aEVecs[:, :nEigen]
    coefsInBOB = rAEVecs.transpose().conjugate().dot(rho0)
    coefsFromDE = splitRealAndImaginaryPartsInVector(v=coefsInBOB)
    # To avoid loosing probability, lets enforce 
    # coefsFromDE[0]=sum(rho0)/surm(\phi_0) (see Note 12 in knigth00)
    coefsFromDE[0] = sum(rho0)/sum(rEVecs[:, 0].real)
    coefsFromDE[1] = 0.0
    coefsColFromDE[:, 0] = coefsFromDE
    prevCoefsFromDE = coefsFromDE

    rhoFromDE = getRealPartOfCArrayDotSRIVector(cArray=rEVecs, sriVector=coefsFromDE)
    rhosFromDECol[:, i] = rhoFromDE

    sigma0Es[0] = sigma0E
    sigma0Is[0] = sigma0I
    spikeRatesFromRho[0] = sigma0E*dv*reversedQs.dot(rho0)
    spikeRatesFromDE[0] = sigma0E*dv*reversedQs.dot(rho0)

    f, axarr = plt.subplots(3)
    axarr[0].set_title("Time %.04f sec" % ts[0])
    axarr[0].plot(vs, rho0, label=r"$\rho$ (full)", color="blue")
    axarr[0].plot(vs, rhoFromDE, label=r"$\rho$ (%d dims)" % nEigen, color="red")
    axarr[0].set_xlabel("Normalized Voltage")
    axarr[0].set_ylabel(r"$\rho$")
    ax0 = axarr[0].twinx()
    ax0.plot(vs, rEVecs[:, eVecToPlot], label=r"$\phi_%d$" % eVecToPlot, 
                 color="green")
    ax0.set_ylabel(r"$\phi$")
    axarr[0].grid()
    axarr[0].legend(loc=2)
    ax0.legend(loc=9)
    axarr[1].plot(ts, spikeRatesFromRho, label="r (full)", color="blue")
    axarr[1].plot(ts, spikeRatesFromDE, label="r (%d dims)" % nEigen, color="red")
    axarr[1].set_xlabel("Time (sec)")
    axarr[1].set_ylabel("Firing Rate (ips)")
    axarr[1].set_xlim(left=t0, right=tf)
    axarr[1].set_ylim(bottom=0, top=50)
    ax1 = axarr[1].twinx()
    ax1.plot(ts, sigma0Es, color="cyan", linestyle="dashed", 
                 label="Input Current")
    ax1.plot(ts, sigma0Es, color="blue", linestyle='dashed', label="Exc. Input")
    ax1.plot(ts, sigma0Is, color="cyan", linestyle='dashed', label="Inh. Input")
    ax1.set_ylabel("Input Current (ips)")
    ax1.set_xlim(left=t0, right=tf)
    ax1.set_ylim(bottom=300, top=1300)
    axarr[1].grid()
    axarr[1].legend(loc=9)
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
        if i%1000==0:
            print("Processing step %d (out of %d)" % (i, len(ts)))
        sigma0E = eStim(t=t)
        sigma0I = iStim(t=t)

        sigmaFEFromRho = (nInputsPerNeuron*fracExcitatoryNeurons*
                           spikeRatesFromRho[i-1])
        sigmaFIFromRho = (nInputsPerNeuron*(1-fracExcitatoryNeurons)*
                           spikeRatesFromRho[i-1])

        sigmaFEFromDE = (nInputsPerNeuron*fracExcitatoryNeurons*
                          spikeRatesFromDE[i-1])
        sigmaFIFromDE = (nInputsPerNeuron*(1-fracExcitatoryNeurons)*
                          spikeRatesFromDE[i-1])

        sigmaEFromRho = sigma0E+sigmaFEFromRho
        sigmaIFromRho = sigma0I+sigmaFIFromRho

        sigmaEFromDE = sigma0E+sigmaFEFromDE
        sigmaIFromDE = sigma0I+sigmaFIFromDE


        rhosDiffMatrix = a0+sigmaEFromRho*a1-sigmaIFromRho*a2
        rho = (identityHD+dt*rhosDiffMatrix).dot(prevRho)
        rhosCol[:, i] = rho
        prevRho = rho

        eVals = eigenRepos.getEigenvalues(sE=sigmaEFromDE, sI=sigmaIFromDE)
        eVecs = eigenRepos.getEigenvectors(sE=sigmaEFromDE, sI=sigmaIFromDE)

        prevREVals = rEVals
        prevREVecs = rEVecs
        rEVals = eVals[:nEigen]
        rEVecs = eVecs[:, :nEigen]

        diagPrevREVals = np.diag(1+dt*prevREVals)
        diffMatrixForDE = np.linalg.pinv(rEVecs).dot(prevREVecs).\
                           dot(diagPrevREVals)
        sriDiffMatrixForDE = \
         splitRealAndImaginaryPartsInMatrix(m=diffMatrixForDE)
        coefsFromDE =  sriDiffMatrixForDE.dot(prevCoefsFromDE)
        # To avoid loosing probability, lets enforce 
        # coefsFromDE[0]=sum(rho0)/surm(\phi_0) (see Note 12 in knigth00)
        coefsFromDE[0] = sum(rho0)/sum(rEVecs[:, 0].real)
        coefsFromDE[1] = 0
        coefsColFromDE[:, i] = coefsFromDE
        prevCoefsFromDE = coefsFromDE

        rhoFromDE = getRealPartOfCArrayDotSRIVector(cArray=rEVecs, sriVector=coefsFromDE)
        rhosFromDECol[:, i] = rhoFromDE

        sigma0Es[i] = sigma0E
        sigma0Is[i] = sigma0I

        sigmaFEsFromRho[i] = sigmaFEFromRho
        sigmaFIsFromRho[i] = sigmaFIFromRho

        sigmaFEsFromDE[i] = sigmaFEFromDE
        sigmaFIsFromDE[i] = sigmaFIFromDE

        sigmaEsFromRho[i] = sigmaEFromRho
        sigmaIsFromRho[i] = sigmaIFromRho

        sigmaEsFromDE[i] = sigmaEFromDE
        sigmaIsFromDE[i] = sigmaIFromDE

        spikeRatesFromRho[i] = sigmaEFromRho*dv*reversedQs.dot(rho)
        spikeRatesFromDE[i] = sigmaEFromDE*dv*reversedQs.dot(rhoFromDE)
        # begin remove
#         print("t=%f: sigmaE=%f, sigmaI=%f" % (t, sigmaEFromDE, sigmaIFromDE))
#         plt.figure()
#         plt.plot(prevREVecs[:, 0])
#         plt.figure()
#         plt.plot(prevREVals)
#         pdb.set_trace()
#         plt.close("all")

#         if t>0.3128799:
#             pdb.set_trace()
#         pdb.set_trace()
        # end remove

        if i%100==0:
            axarr[0].cla()
            axarr[0].set_title("Time %.04f sec" % ts[i])
            axarr[0].plot(vs, rho, label=r"$\rho$ (full)", color="blue")
            axarr[0].plot(vs, rhoFromDE, label=r"$\rho$ (%d dims)" % nEigen,
                              color="red")
            axarr[0].set_xlabel("Normalized Voltage")
            axarr[0].set_ylabel(r"$\rho$")

            ax0.cla()
            ax0.plot(vs, rEVecs[:, eVecToPlot], label=r"$\phi_%d$" % eVecToPlot, 
                         color="green")
            ax0.set_ylabel(r"$\phi$")
            axarr[0].grid()
            axarr[0].legend(loc=2)
            ax0.legend(loc=9)
            axarr[1].cla()
            axarr[1].plot(ts, spikeRatesFromRho, label="r (full)", color="blue")
            axarr[1].plot(ts, spikeRatesFromDE, label="r (%d dims)" % nEigen,
                              color="red")
            axarr[1].set_xlabel("Time (sec)")
            axarr[1].set_ylabel("Firing Rate (ips)")
            axarr[1].set_ylim(bottom=0, top=50)
            ax1.cla()

            ax1.plot(ts, sigma0Es, color="blue", linestyle='dashed', 
                         label="Exc. Input")
            ax1.plot(ts, sigma0Is, color="cyan", linestyle='dashed', 
                         label="Inh. Input")
            ax1.set_xlim(left=t0, right=tf)
            ax1.set_ylim(bottom=300, top=1300)
            axarr[1].grid()
            axarr[1].legend(loc=9)
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
#     np.savez(resultsFilename, ts=ts, vs=vs, rhosCol=rhosCol, rhosFromDECol=rhosFromDECol, coefsColFromDE=coefsColFromDE, spikeRatesFromRho=spikeRatesFromRho, spikeRatesFromDE=spikeRatesFromDE)

if __name__ == "__main__":
    main(sys.argv)

