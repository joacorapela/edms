
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from LLsPlotter import LLsPlotter

def main(argv):
    nLevels=60
    llsFilename = \
     "results/llsOnePopulationT00.00Tf0.25Dt0.000010Leakage4.00-41.00-4.00NInputsPerNeuron0-19-2FracExcitatoryNeurons0.00-0.45-0.10.npz"
    ysFilename = \
     "results/ysForTwoPopulationsT00.00Tf0.25Dt0.000010Leakage20.00G10.00f0.20Sigma2.00.npz"
    llsFigFilenamePattern = \
     "figures/llsOnePopulationT00.00Tf0.25Dt0.000010Leakage4.00-41.00-4.00NInputsPerNeuron0-19-2FracExcitatoryNeurons0.00-0.45-0.10-X%s-Y%s.eps"

    loadRes = np.load(llsFilename)
    params = loadRes["params"]
    lls = loadRes["lls"]

    loadRes = np.load(ysFilename)
    trueParams = loadRes["params"][0]

    llsPlotter = LLsPlotter()

    paramXIndex = 1
    paramYIndex = 0
    llsFigFilename = llsFigFilenamePattern % ("G", "Leakage")
    llsPlotter.plotLLs(lls=lls, params=params, trueParams=trueParams,
                              paramXIndex=paramXIndex, paramYIndex=paramYIndex,
                              xlabel="G", ylabel="Leakage", 
                              figFilename=llsFigFilename, nLevels=nLevels)

    paramXIndex = 2
    paramYIndex = 0
    llsFigFilename = llsFigFilenamePattern % ("f", "Leakage")
    llsPlotter.plotLLs(lls=lls, params=params, trueParams=trueParams,
                              paramXIndex=paramXIndex, paramYIndex=paramYIndex,
                              xlabel="f", ylabel="Leakage", 
                              figFilename=llsFigFilename, nLevels=nLevels)

    paramXIndex = 1
    paramYIndex = 2
    llsFigFilename = llsFigFilenamePattern % ("G", "f")
    llsPlotter.plotLLs(lls=lls, params=params, trueParams=trueParams,
                              paramXIndex=paramXIndex, paramYIndex=paramYIndex,
                              xlabel="G", ylabel="f", 
                              figFilename=llsFigFilename, nLevels=nLevels)

if __name__ == "__main__":
    main(sys.argv)

