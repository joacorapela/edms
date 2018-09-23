
import sys
import numpy as np
import math
import pickle
import pdb
import matplotlib.pyplot as plt
from ifEDMsFunctions import computeA0, computeA1, computeA2, computeQRs
from TwoPopulationsIFEnsembleDensitySimpleIntegrator import TwoPopulationsIFEnsembleDensitySimpleIntegrator

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    eRho0 = rho0
    iRho0 = rho0
    t0 = 0.0
#     tf = 1.0
    tf = 0.30
    dt = 1e-5
    wIE = 15.0
    wEI = 50.0
    sigma = 2.0
    xticksSep = 0.1
    resultsFilenamePattern = "%s/simpleYsForTwoPopulationsT0%.02fTf%.02fDt%fWEI%.02fWIE%.02fSigma%.02f.%s"
    resultsFilename = resultsFilenamePattern % ("results", t0, tf, dt, wEI, wIE, sigma, "npz")
    figureFilename = resultsFilenamePattern % ("figures", t0, tf, dt, wEI, wIE, sigma, "eps")

    def addUncorrelatedGaussianNoiseToSpikesRatesForParams(srsForPops, sigma):
        nsrsForPops = np.empty(srsForPops.shape)
        for j in xrange(srsForPops.shape[1]):
            srsForPop = srsForPops[:, j]
            nsrsForPops[:, j] = srsForPop + np.random.normal(loc=0, 
                                                              scale=sigma,
                                                              size=
                                                               srsForPop.size)
        return(nsrsForPops)

    def eInput(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    times = np.arange(start=t0, stop=tf+dt, step=dt)
    eInputs = eInput(t=times)

    a0 = computeA0(nVSteps=nVSteps, leakage=leakage)
    a1 = computeA1(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu, kappaSigma=kappaSigma)
    reversedQs = computeQRs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)

    integrator = TwoPopulationsIFEnsembleDensitySimpleIntegrator(a0=a0, 
                                                                  a1=a1,
                                                                  a2=a2, 
                                                                  dt=dt,
                                                                  dv=dv, 
                                                                  reversedQs=
                                                                   reversedQs)
    _, _, eRs, iRs = integrator.integrate(eRho0=eRho0, iRho0=iRho0,
                                           wEI=wEI, wIE=wIE, 
                                           eInputs=eInputs)
    srsForPops = np.column_stack((eRs, iRs))
    ys = addUncorrelatedGaussianNoiseToSpikesRatesForParams(srsForPops=
                                                              srsForPops, 
                                                             sigma=sigma)
    np.savez(resultsFilename, sigma=sigma, params=[(wEI, wIE)], ys=ys,
                              rs=srsForPops)

    plt.close('all')
    myXTicks = np.arange(t0, tf+dt, xticksSep)
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(times, ys[:, 0], label="Measurements")
    ax1.plot(times, srsForPops[:, 0], label="EDM spike rates", color="red")
    ax1.grid()
    ax1.set_xticks(myXTicks)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Firing Rate (ips)")
    ax1.set_title("Excitatory Ensemble")

    ax2.plot(times, ys[:, 1], label="Mesurements")
    ax2.plot(times, srsForPops[:, 1], label="EDM spike rates", color="red")
    ax2.grid()
    ax2.set_xticks(myXTicks)
    ax2.legend()
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Inhibitory Ensemble")

    plt.savefig(figureFilename)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

