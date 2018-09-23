
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from LLsPlotter import LLsPlotter

def main(argv):
    t0 = 0.00
    dt = 1e-5
    tf = 0.25
    tfSaveYs = 0.3
    ysSigma = 2.0

    wEIsStart = 10
    wEIsStop = 100
    wEIsStep = 10
    wIEsStart = 5
    wIEsStop = 50
    wIEsStep = 5

    trueWEI = 50.0
    trueWIE = 15.0

    llsFilenamePattern = "%s/twoPopulationsSimple2LLsT0%.02fTf%.02fDt%fTrueWEI%.02fTrueWIE%.02fWEI%.02f-%.02f-%.02fWIE%.02f-%.02f-%.02fSigma%.02f.%s"
    llsFilename = llsFilenamePattern % ("results", t0, tf, dt, trueWEI, trueWIE, wEIsStart, wEIsStop, wEIsStep, wIEsStart, wIEsStop, wIEsStep, ysSigma, "npz")
    llsFigFilename = llsFilenamePattern % ("figures", t0, tf, dt, trueWEI, trueWIE, wEIsStart, wEIsStop, wEIsStep, wIEsStart, wIEsStop, wIEsStep, ysSigma, "ll.eps")
    ysFilename = "results/simple2YsForTwoPopulationsT0%.02fTf%.02fDt%fWEI%.02fWIE%.02fSigma%.02f.npz" % (t0, tfSaveYs, dt, trueWEI, trueWIE, ysSigma)

    paramXIndex = 1
    paramYIndex = 0

    loadRes = np.load(llsFilename)
    params = loadRes["params"]
    lls = loadRes["lls"]

    loadRes = np.load(ysFilename)
    trueParams = loadRes["params"][0]

    llsPlotter = LLsPlotter()
    llsPlotter.plotLLs(lls=lls, params=params, trueParams=trueParams,
                              paramXIndex=paramXIndex, paramYIndex=paramYIndex,
                              xlabel=r"$W_{ie}$", ylabel=r"$W_{ei}$", 
                              figFilename=llsFigFilename, nLevels=30)

if __name__ == "__main__":
    main(sys.argv)

