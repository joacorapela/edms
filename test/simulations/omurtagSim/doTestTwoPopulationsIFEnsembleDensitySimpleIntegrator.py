
import sys
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
from ifEDMsFunctions import computeA0, computeA1, computeA2, computeQRs
from TwoPopulationsIFEnsembleDensitySimpleIntegrator import \
 TwoPopulationsIFEnsembleDensitySimpleIntegrator

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    eNInputsPerNeuron = 10
    iNInputsPerNeuron = 10
    eFracExcitatoryNeurons = 0.2
    iFracExcitatoryNeurons = 0.2

    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    eRho0 = rho0
    iRho0 = rho0

    wEI = 50
    wIE = 15

    t0 = 0.0
    tf = 0.25
    dt = 1e-5
    # resultsFilename = 'results/twoPSinusoidalSimpleIntegrationResult.npz'

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
    eRhos, iRhos, eRs, iRs = integrator.integrate(eRho0=eRho0, iRho0=iRho0,
                                                   wEI=wEI, wIE=wIE, 
                                                   eInputs=eInputs)
    # np.savez(resultsFilename, times=times, eRhos=eRhos, iRhos=iRhos, eRs=eRs, iRs=iRs)
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(times, eRs)
    ax1.set_title("Excitatory Spike Rates")
    ax2.plot(times, iRs)
    ax2.set_title("Inhibitory Spike Rates")
    plt.show()
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

