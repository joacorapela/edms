
import sys
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
from ifEDMsFunctions import computeA0, computeA1, computeA2, computeQRs
from IFEnsembleDensitySimpleIntegratorRNFNI import IFEnsembleDensitySimpleIntegratorRNFNI

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    t0 = 0.0
    tf = 0.25
    dt = 1e-5

    def input(t, sigma0=800, b=0.5, omega=6*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    times = np.arange(start=t0, stop=tf+dt, step=dt)
    inputs = input(t=times)

    a0 = computeA0(nVSteps=nVSteps, leakage=leakage)
    a1 = computeA1(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    reversedQs = computeQRs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)

    integrator = IFEnsembleDensitySimpleIntegratorRNFNI(a0=a0, a1=a1, dt=dt, 
                                                               dv=dv, 
                                                               reversedQs=
                                                                reversedQs)
    rhos, rs = integrator.integrate(rho0=rho0, inputs=inputs)

    plt.close('all')
    plt.plot(times, rs)
    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate (ips)")

    plt.show()

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

