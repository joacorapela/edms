
import sys
import numpy as np
import math
import pdb
from IFEnsembleDensityIntegratorRNFWI import IFEnsembleDensityIntegratorRNFWI 
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI 
from TwoPopulationsIFEDMsGradientCalculator import TwoPopulationsIFEDMsGradientCalculator
from TwoPopulationsIFEDMsMaximumLikelihoodOptimizer import TwoPopulationsIFEDMsMaximumLikelihoodOptimizer

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
    eEDMRho0 = rho0
    iEDMRho0 = rho0

    t0 = 0.0
    tf = 0.25
    tfSaveYs = 0.4
    dt = 1e-5
    dtSaveRhos = 1e-5
    wIEs = np.array([15])
    wEIs = np.array([45, 55])
    wEE = None
    wIE = 15
    wEI = 50
    wII = None
    ysSigma = 0.01
#     ysSigma = 1e-6
    ysFilename = "results/ysForTwoPopulationsT0%.02fTf%.02fDt%fWEE%sWIE%sWEI%sWII%sSigma%.02f.npz" % (t0, tfSaveYs, dt, str(wEE), str(wIE), str(wEI), str(wII), ysSigma)
    resultsFilename = "results/twoPopulationsMLGradientsSigma%.02f.npz" % (ysSigma)

    ifEDIntegrator1 = \
     IFEnsembleDensityIntegratorRNFWI(nVSteps=nVSteps, 
                                       leakage=leakage, 
                                       hMu=hMu, hSigma=hSigma, 
                                       kappaMu=kappaMu, 
                                       kappaSigma=kappaSigma)
    ifEDIntegrator2 = \
     IFEnsembleDensityIntegratorRNFNI(nVSteps=nVSteps,
                                       leakage=leakage, 
                                       hMu=hMu, hSigma=hSigma)

    def sinusoidalInputMeanFrequency(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    npLoadRes = np.load(ysFilename)
    ys = npLoadRes["ys"]
    ysSaveRhosIndices = range(0, int(round(tf/dt))+1, int(round(dtSaveRhos/dt)))
    y1s = ys[ysSaveRhosIndices, 0]
    y2s = ys[ysSaveRhosIndices, 1]
    mlOptimizer = TwoPopulationsIFEDMsMaximumLikelihoodOptimizer( \
                   ifEDIntegrator1=ifEDIntegrator1, 
                   ifEDIntegrator2=ifEDIntegrator2, 
                   sigma0=sinusoidalInputMeanFrequency, 
                   edm1Rho0=eEDMRho0, 
                   edm2Rho0=iEDMRho0,
                   t0=t0, tf=tf, dt=dt, dv=dv, dtSaveRhos=dtSaveRhos, 
                   nVSteps=nVSteps)
    gradientCalculator = TwoPopulationsIFEDMsGradientCalculator( \
                   a0=ifEDIntegrator1._a0, 
                   a1=ifEDIntegrator1._a1, 
                   a2=ifEDIntegrator1._a2, 
                   sigma0=sinusoidalInputMeanFrequency, 
                   dt=dtSaveRhos,
                   dv=dv,
                   reversedQs=ifEDIntegrator1._reversedQs)
    derivs = np.empty((wEIs.size*wIEs.size, 4))
    rowIndex = 0
    for wEI in wEIs:
        for wIE in wIEs:
            times1, r1s, rho1s, times2, r2s, rho2s = \
             mlOptimizer._integrateEDMs(w12=wEI, w21=wIE)
            dWEI, dWIE = gradientCalculator.deriv(w12=wEI, w21=wIE,
                                                           times1=times1,
                                                           times2=times2,
                                                           y1s=y1s, y2s=y2s,
                                                           r1s=r1s, r2s=r2s,
                                                           rho1s=rho1s, rho2s=rho2s)
            derivs[rowIndex, 0] = wEI
            derivs[rowIndex, 1] = wIE
            derivs[rowIndex, 2] = dWEI
            derivs[rowIndex, 3] = dWIE
            rowIndex = rowIndex + 1
    np.savez(resultsFilename, derivs=derivs)

if __name__ == "__main__":
    main(sys.argv)

