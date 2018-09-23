
import sys
import numpy as np
import math
import pdb
from ifEDMsFunctions import computeA0, computeA1, computeA2, computeQRs
from TwoPopulationsIFEDMsMaximumLikelihoodSimpleOptimizer import TwoPopulationsIFEDMsMaximumLikelihoodSimpleOptimizer

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
    tf = 0.3
    tfSaveYs = 0.3
    dt = 1e-5
    wEI0 = 37
    wIE0 = 3
    stepSize=1e-3
    tol=2e2
    maxIter=1000
    trueWEI = 50.0
    trueWIE = 15.0
    ysSigma = 2.0
    ysFilename = "results/simpleYsForTwoPopulationsT0%.02fTf%.02fDt%fWEI%.02fWIE%.02fSigma%.02f.npz" % (t0, tfSaveYs, dt, trueWEI, trueWIE, ysSigma)
    resultsFilename = "results/twoPopulationsSimpleMLEstimatesT0%.02fTf%.02fDt%fTrueWEI%.02fTrueWIE%.02fWEI0%.02fWIE0%.02fStepSize%fTol%fMaxIter%dYsSigma%.02f.npz" % (t0, tf, dt, trueWEI, trueWIE, wEI0, wIE0, stepSize, tol, maxIter, ysSigma)

    times = np.arange(start=t0, stop=tf+dt, step=dt)
    def eInput(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    eInputs = eInput(t=times)

    a0 = computeA0(nVSteps=nVSteps, leakage=leakage)
    a1 = computeA1(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu, kappaSigma=kappaSigma)
    reversedQs = computeQRs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)

    mlOptimizer = TwoPopulationsIFEDMsMaximumLikelihoodSimpleOptimizer( \
                   a0=a0, a1=a1, a2=a2, dt=dt, dv=dv, reversedQs=reversedQs)

    npLoadRes = np.load(ysFilename)
    ys = npLoadRes["ys"]
    eYs = ys[:, 0]
    iYs = ys[:, 1]

    wEI, wIE, converged, ws, lls, normGradients = \
     mlOptimizer.optimize(eYs=eYs, iYs=iYs, wEI0=wEI0, wIE0=wIE0, 
                                   eRho0=eRho0, iRho0=iRho0, eInputs=eInputs,
                                   ysSigma2=ysSigma**2, stepSize=stepSize,
                                   tol=tol, maxIter=maxIter)

    np.savez(resultsFilename, wEI=wEI, wIE=wIE, convered=converged, ws=ws,
                              lls=lls, normGradients=normGradients)

if __name__ == "__main__":
    main(sys.argv)

