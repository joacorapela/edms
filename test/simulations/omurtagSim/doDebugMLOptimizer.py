
import sys
import numpy as np
import numpy.linalg as la
import math
import pdb
from IFEnsembleDensityIntegratorRNFWI import IFEnsembleDensityIntegratorRNFWI 
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI 
from TwoPopulationsIFEDMsMaximumLikelihoodOptimizer import TwoPopulationsIFEDMsMaximumLikelihoodOptimizer

def main(argv):

    def geneateData(phi10, phi20, dt, a0, a1, a2, sigma0, reversedQs, w12, w21):
        idM = np.identity(a0.shape[0])

        r10 = dv*sigma0(t=0)*reversedQs.dot(phi10)
        r20 = 0.0
        invR10 = la.inv(idM-dt*(a0+sigma0(t=dt)*a1-w21*r2[0]*a2)
        phi11 = invR10.dot(phi0) 

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
    vs = (np.linspace(1, nVSteps, nVSteps))/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    eEDMRho0 = rho0
    iEDMRho0 = rho0

    t0 = 0.0
    tf = 0.25
    tfSaveYs = 0.4
    dt = 1e-5
    dtSaveRhos = 1e-3
    wEI0 = 42
    wIE0 = 12
#     wEI0 = 50
#     wIE0 = 15
    stepSize=1e-3
    tol=1e-4
    maxIter=1000
    wEE = None
    wIE = 15
    wEI = 50
    wII = None
    ysSigma = 5.00

    r1s, r2s, phi1s, phi2s = generateData(phi10=eEDMRho0, phi20=iEDMRho0,
                                                          dt=dt,
                                                          a0=ifEDIntegrator._a0,
                                                          a1=ifEDIntegrator._a1,
                                                          a2=ifEDIntegrator._a2,
                                                          sigma0=sinusoidalInputMeanFrequency,
                                                          reversedQs=
                                                           ifEDIntegrator._reversedQs)
    dLLW12, dLLW21 = getGradient(r1s=r1s, r2s=r2s, phi1s=phi1s,
                                                phi2s=phi2s)
#
#     ysSigma = 1e-6
    ysFilename = "results/ysForTwoPopulationsT0%.02fTf%.02fDt%fEEDMEW%sEEDMIW%sIEDMEW%sIEDMIW%sSigma%.02f.npz" % (t0, tfSaveYs, dt, str(wEE), str(wIE), str(wEI), str(wII), ysSigma)
    resultsFilename = "results/twoPopulationsMLEstimatesT0%.02fTf%.02fDt%fDtSaveRhos%fW120%.02fW210%.02fStepSize%fTol%fMaxIter%dEEDMEW%sEEDMIW%sIEDMEW%sIEDMIW%sYsSigma%.02f.npz" % (t0, tf, dt, dtSaveRhos, wEI0, wIE0, stepSize, tol, maxIter, str(wEE), str(wIE), str(wEI), str(wII), ysSigma)

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

    mlOptimizer = TwoPopulationsIFEDMsMaximumLikelihoodOptimizer( \
                   ifEDIntegrator1=ifEDIntegrator1, 
                   ifEDIntegrator2=ifEDIntegrator2, 
                   sigma0=sinusoidalInputMeanFrequency, 
                   edm1Rho0=eEDMRho0, 
                   edm2Rho0=iEDMRho0,
                   t0=t0, tf=tf, dt=dt, dtSaveRhos=dtSaveRhos, 
                   nVSteps=nVSteps)
    npLoadRes = np.load(ysFilename)
    ys = npLoadRes["ys"]
    ysSaveRhosIndices = range(0, int(round(tf/dt))+1, int(round(dtSaveRhos/dt)))
    y1s = ys[ysSaveRhosIndices, 0]
    y2s = ys[ysSaveRhosIndices, 1]
    w12, w21, converged, ws, mses = \
     mlOptimizer.optimize(y1s=y1s, y2s=y2s, w120=wEI0, w210=wIE0, 
                                   stepSize=stepSize, tol=tol, maxIter=maxIter)
    np.savez(resultsFilename, w12=w12, w21=w21, convered=converged, ws=ws,
                              mses=mses)

if __name__ == "__main__":
    main(sys.argv)

