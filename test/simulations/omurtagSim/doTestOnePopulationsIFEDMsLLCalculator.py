
import sys
import numpy as np
import math
import pdb
from ifEDMsFunctions import computeA0, computeA1, computeA2, computeQRs
from TwoPopulationsIFEDMsSimpleGradientCalculator import TwoPopulationsIFEDMsSimpleGradientCalculator
from TwoPopulationsIFEnsembleDensitySimpleIntegrator import TwoPopulationsIFEnsembleDensitySimpleIntegrator
from OnePopulationLLCalculator import OnePopulationLLCalculator

def main(argv):

    spikesRatesForParamsFilename = \
     "results/spikeRatesForParamsOfOnePopulationT00.00Tf0.25Dt0.000010Leakage4.00-41.00-4.00NInputsPerNeuron0-19-2FracExcitatoryNeurons0.00-0.45-0.10.npz"
    ysFilename = \
     "results/ysForTwoPopulationsT00.00Tf0.25Dt0.000010Leakage20.00G10.00f0.20Sigma2.00.npz"
    resultsFilename= \
     "results/llsOnePopulationT00.00Tf0.25Dt0.000010Leakage4.00-41.00-4.00NInputsPerNeuron0-19-2FracExcitatoryNeurons0.00-0.45-0.10.npz"

    loadRes = np.load(spikesRatesForParamsFilename)
    params = loadRes["params"]
    spikeRatesForParams = loadRes["spikeRates"]
    
    loadRes = np.load(ysFilename)
    ys = loadRes["ys"]
    sigma = loadRes["sigma"]
    trueParams = loadRes["params"]

    llCalculator = OnePopulationLLCalculator()
    lls = llCalculator.calculateLLsForParams(ys=ys, 
                                        spikeRatesForParams=
                                         spikeRatesForParams, 
                                        sigma=sigma)
    np.savez(resultsFilename, params=params, lls=lls)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

