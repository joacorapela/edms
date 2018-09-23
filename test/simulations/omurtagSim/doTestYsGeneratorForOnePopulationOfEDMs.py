
import sys
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
from IFEnsembleDensityIntegratorRWFWI import IFEnsembleDensityIntegratorRWFWI 
from SpikeRatesForParamsCalculatorOfOnePopulationOfEDMs import \
 SpikeRatesForParamsCalculatorOfOnePopulationOfEDMs
import pickle

def main(argv):
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    nInputsPerNeuron = 10
    fracExcitatoryInputs = 0.2
    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    rho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(vs-mu)**2/(2*sigma2))
    t0 = 0.0
#     tf = 1.0
    tf = 0.25
    dt = 1e-5
    sigma = 2.0
    resultsFilename = "results/ysForTwoPopulationsT0%.02fTf%.02fDt%fLeakage%.02fG%.02ff%.02fSigma%.02f.npz" % (t0, tf, dt, leakage, nInputsPerNeuron, fracExcitatoryInputs, sigma)

    def eInputSigma(t, sigma0=800, b=0.5, omega=8*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    def iInputSigma(t, sigma0=200, b=0.5, omega=4*np.pi):
        return(sigma0*(1+b*np.sin(omega*t)))

    spikeRatesForParamsCalculator = \
     SpikeRatesForParamsCalculatorOfOnePopulationOfEDMs(\
      hMu=hMu,
      hSigma=hSigma,
      kappaMu=kappaMu,
      kappaSigma=kappaSigma,
      t0=t0, 
      tf=tf, 
      dt=dt,
      nVSteps=nVSteps,
      rho0=rho0,
      eInputSigma=eInputSigma,
      iInputSigma=iInputSigma)
    params = [(leakage, nInputsPerNeuron, fracExcitatoryInputs)]
    times, srsForParams = spikeRatesForParamsCalculator.\
     calculateSpikeRatesForParams(params=params)
#     with open("spikeRatesForParams.pickle", 'wb') as f:
#         pickle.dump((times, srsForParams), f)
#     with open("spikeRatesForParams.pickle", 'rb') as f:
#         (times, srsForParams) = pickle.load(f)
    srs = srsForParams[0, :]
    ys = srs + np.random.normal(loc=0, scale=sigma, size=srs.size)

    np.savez(resultsFilename, sigma=sigma, params=params, ys=ys)
    plt.plot(times, ys, label="noisy", color="blue")
    plt.plot(times, srs, label="noiseless", color="red")
    plt.legend()
    plt.show()
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

