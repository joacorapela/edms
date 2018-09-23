
import sys
import numpy as np
import math
import pdb
from ifEDMsFunctions import computeA0, computeA1, computeA2, computeQRs
from TwoPopulationsIFEDMsSimpleGradientCalculator import TwoPopulationsIFEDMsSimpleGradientCalculator
from TwoPopulationsIFEnsembleDensitySimpleIntegrator import TwoPopulationsIFEnsembleDensitySimpleIntegrator
from TwoPopulationsLLCalculator import TwoPopulationsLLCalculator

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

    wEIsStart = 35
    wEIsStop = 70
    wEIsStep = 5
    wEIs = np.arange(start=wEIsStart, stop=wEIsStop, step=wEIsStep)
    wIEsStart = 0
    wIEsStop = 35
    wIEsStep = 5
    wIEs = np.arange(start=wIEsStart, stop=wIEsStop, step=wIEsStep)

    trueWEI = 50.0
    trueWIE = 15.0
    ysSigma = 2.0
#     ysSigma = 1e-6
    ysFilename = "results/simpleYsForTwoPopulationsT0%.02fTf%.02fDt%fWEI%.02fWIE%.02fSigma%.02f.npz" % (t0, tfSaveYs, dt, trueWEI, trueWIE, ysSigma)
    resultsFilename = "results/twoPopulationsMLSimpleLLsAndGradientsT0%.02fTf%.02fDt%fTrueWEI%.02fTrueWIE%.02fWEI%.02f-%.02f-%.02fWIE%.02f-%.02f-%.02fSigma%.02f.npz" % (t0, tf, dt, trueWEI, trueWIE, wEIsStart, wEIsStop, wEIsStep, wIEsStart, wIEsStop, wIEsStep, ysSigma)

    times = np.arange(start=t0, stop=tf+dt, step=dt)
    def eInput(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    eInputs = eInput(t=times)

    a0 = computeA0(nVSteps=nVSteps, leakage=leakage)
    a1 = computeA1(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu, kappaSigma=kappaSigma)
    reversedQs = computeQRs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)

    npLoadRes = np.load(ysFilename)
    ys = npLoadRes["ys"]
    eYs = ys[:, 0]
    iYs = ys[:, 1]

    twoPIntegrator = \
     TwoPopulationsIFEnsembleDensitySimpleIntegrator(a0=a0, a1=a1, a2=a2, 
                                                            dt=dt, dv=dv, 
                                                            reversedQs=
                                                             reversedQs)
    gradientCalculator = \
     TwoPopulationsIFEDMsSimpleGradientCalculator(a0=a0, a1=a1, a2=a2, 
                                                         eInputs=eInputs, 
                                                         dt=dt, dv=dv, 
                                                         reversedQs=reversedQs,
                                                         sigma2=ysSigma**2)
    llCalculator = TwoPopulationsLLCalculator()
    llsAndGradients = np.empty((wEIs.size*wIEs.size, 5))
    rowIndex = 0
    for wEI in wEIs:
        for wIE in wIEs:
            print("Processing wEI=%.02f and wIE=%.02f" % (wEI, wIE))
            sys.stdout.flush()
            eRhos, iRhos, eRs, iRs = \
             twoPIntegrator.integrate(eRho0=eRho0, iRho0=iRho0,
                                                   wEI=wEI, wIE=wIE, 
                                                   eInputs=eInputs)
            dWEI, dWIE = gradientCalculator.deriv(w12=wEI, w21=wIE,
                                                           y1s=eYs, y2s=iYs,
                                                           r1s=eRs, r2s=iRs,
                                                           rho1s=eRhos, 
                                                           rho2s=iRhos, 
                                                           eInputs=eInputs)
            ll = llCalculator.calculateLL(eYs=eYs, iYs=iYs, eRs=eRs, iRs=iRs, 
                                                   sigma2=ysSigma**2)
            llsAndGradients[rowIndex, 0] = wEI
            llsAndGradients[rowIndex, 1] = wIE
            llsAndGradients[rowIndex, 2] = dWEI
            llsAndGradients[rowIndex, 3] = dWIE
            llsAndGradients[rowIndex, 4] = ll
            rowIndex = rowIndex + 1
    np.savez(resultsFilename, llsAndGradients=llsAndGradients)

if __name__ == "__main__":
    main(sys.argv)

