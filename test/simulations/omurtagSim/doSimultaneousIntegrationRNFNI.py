
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle
from myUtils import plotSpikeRates, plotRhos
from ifEDMsFunctions import computeQs
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI
from EigenReposForOneStim import EigenReposForOneStim

def main(argv):
    plt.ion()

    if(len(argv)>1):
        nEigen = int(argv[1])
    else:
        nEigen = 17

#     resultsFilename = "results/simultaneousIntegration3.npz"
    eigenReposFilename = "results/anEigenResposForOneStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571StimFrom300.000000StimTo1300.000000StimStep20.000000NEigen17.pickle"
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
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

    sigma0 = 800

    b = 0.6
    freq = 4
    def sinusoidalInputMeanFrequency(t, sigma0=sigma0, b=b, omega=2*np.pi*freq):
        return(sigma0*(1+b*np.sin(omega*t)))

    def constantInputMeanFrequency(t, sigma0=sigma0):
        return(sigma0)

    linearStimSlope = 1
    def linearInputMeanFrequency(t, sigma0=sigma0, slope=linearStimSlope):
        return(sigma0+t*linearStimSlope)

    stim = sinusoidalInputMeanFrequency

#     stim = constantInputMeanFrequency

#     stim = linearInputMeanFrequency

    minStim = sigma0*(1-b)
    maxStim = sigma0*(1+b)
#     maxStim = 325
#     nStimSteps = 100
#     stepStim = (maxStim-minStim)/nStimSteps
    stepStim = 4.0


    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRNFNI(nVSteps=nVSteps, leakage=leakage, 
                                                       hMu=hMu, hSigma=hSigma)
    with open(eigenReposFilename, 'rb') as f:
        eigenRepos = pickle.load(f)
    a0 = ifEDIntegrator._a0
    a1 = ifEDIntegrator._a1

    ts = np.arange(t0, tf, dt)
    rhosCol = np.empty([rho0.size, ts.size])
    rhosFromDECol = np.empty([rho0.size, ts.size], dtype=complex)
    coefsColFromDE = np.empty([nEigen, ts.size], dtype=complex)
    sigmaEs = np.empty(ts.size)
    spikeRatesFromRhos = np.empty(ts.size)
    spikeRatesFromDE = np.empty(ts.size, dtype=complex)
    sigmaEs[:] = np.nan
    spikeRatesFromRhos[:] = np.nan
    spikeRatesFromDE[:] = np.nan

    sigmaE = sinusoidalInputMeanFrequency(t=0)
#     sigmaE = stepInputMeanFrequency(t=0)
    rhosDiffMatrix = a0+sigmaE*a1
    eVals = eigenRepos.getEigenvalues(s=sigmaE)
    eVecs = eigenRepos.getEigenvectors(s=sigmaE)
    aEVecs = np.linalg.pinv(eVecs).transpose().conjugate()
    rEVals = eVals[:nEigen]
    rEVecs = eVecs[:, :nEigen]
    rAEVecs = aEVecs[:, :nEigen]
    coefsInBOB = rAEVecs.transpose().conjugate().dot(rho0)
    coefsColFromDE[:, 0] = coefsInBOB

    sigmaEs[0] = sigmaE
    spikeRatesFromRhos[0] = sigmaE*dv*reversedQs.dot(rho0)
    spikeRatesFromDE[0] = sigmaE*dv*coefsInBOB.\
                            dot(rEVecs.transpose().dot(reversedQs))

    rhos0FromBOB = rEVecs.dot(coefsInBOB)
    rhosCol[:, 0] = rho0
    rhosFromDECol[:, 0] = rhos0FromBOB

    f, axarr = plt.subplots(3)
    axarr[0].set_title("Time %.04f sec" % ts[0])
    axarr[0].plot(vs, rho0, label="Full", color="blue")
    axarr[0].plot(vs, rhos0FromBOB.real, label="%d dims" % nEigen, color="red")
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
    axarr[2].plot(coefsInBOB, color="red")
    axarr[2].set_xlabel("Low-Dimensional Coefficient Index")
    axarr[2].set_ylabel("Coefficient Value")
    axarr[2].grid()
    axarr[2].legend(loc=1)

    plt.draw()
#     pdb.set_trace()

    prevRho = rho0
    prevCoefsFromDE = coefsInBOB
    identityHD = np.identity(a0.shape[0])
    identityLD = np.identity(nEigen)
    for i in xrange(1, len(ts)):
        t = ts[i]
        if i%100==0:
            print("Processing step %d (out of %d)" % (i, len(ts)))
        sigmaE = stim(t=t)
#         sigmaE = stepInputMeanFrequency(t=t)

        rhosDiffMatrix = a0+sigmaE*a1
        rho = (identityHD+dt*rhosDiffMatrix).dot(prevRho)
        rhosCol[:, i] = rho
        prevRho = rho

        eVals = eigenRepos.getEigenvalues(s=sigmaE)
        eVecs = eigenRepos.getEigenvectors(s=sigmaE)

        prevREVecs = rEVecs
        rEVals = eVals[:nEigen]
        rEVecs = eVecs[:, :nEigen]

        diagCoefs = 1+dt*rEVals
        diffMatrixForDE = np.linalg.pinv(rEVecs).dot(prevREVecs).\
                        dot(np.diag(diagCoefs))
        coefsFromDE =  diffMatrixForDE.dot(prevCoefsFromDE)
        # To avoid loosing probability, lets enforce 
        # coefsFromDE[0]=sum(rho0)/surm(\phi_0) (see Note 12 in knigth00)
        coefsFromDE[0] = sum(rho0)/sum(rEVecs[:, 0])
        coefsColFromDE[:, i] = coefsFromDE
        prevCoefsFromDE = coefsFromDE

        sigmaEs[i] = sigmaE
        spikeRatesFromRhos[i] = sigmaE*dv*reversedQs.dot(rho)
        spikeRatesFromDE[i] = sigmaE*dv*coefsFromDE.\
                                dot(rEVecs.transpose().\
                                     dot(reversedQs))
        rhoFromDE = rEVecs.dot(coefsFromDE)
        rhosFromDECol[:, i] = rhoFromDE

        if i%100==0:
            axarr[0].cla()
            axarr[0].set_title("Time %.04f sec" % ts[i])
            axarr[0].plot(vs, rho, label="Full", color="blue")
            axarr[0].plot(vs, rhoFromDE.real, label="%d dims" % nEigen,
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
            ax1.plot(ts, sigmaEs, color="cyan", linestyle='dashed', 
                         label="Input Current")
            ax1.set_ylabel("Input Current (ips)")
            ax1.set_ylim(bottom=300, top=1300)
            ax1.legend(loc=1)

            axarr[2].cla()
            axarr[2].plot(coefsFromDE, color="red")
            axarr[2].set_xlabel("Low-Dimensional Coefficient Index")
            axarr[2].set_ylabel("Coefficient Value")
            axarr[2].grid()
            axarr[2].legend(loc=8)

            plt.draw()
#             pdb.set_trace()

#         pdb.set_trace()
    pdb.set_trace()
#     np.savez(resultsFilename, ts=ts, vs=vs, rhosCol=rhosCol, rhosFromDECol=rhosFromDECol, coefsColFromDE=coefsColFromDE, spikeRatesFromRhos=spikeRatesFromRhos, spikeRatesFromDE=spikeRatesFromDE)

if __name__ == "__main__":
    main(sys.argv)

