
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from LLCalculator import LLCalculator

def main(argv):
    t0 = 0.00
    dt = 1e-5
    tf = 0.3
    ysSigma = 2.0

    wEI0 = 37
    wIE0 = 3
    stepSize=1e-3
    tol=2e2
    maxIter=1000
    trueWEI = 50.0
    trueWIE = 15.0

    minLLToPlot = -100000000
    nLevels = 30
    arrowsColor = "white"

    wEIsStart = 35
    wEIsStop = 70
    wEIsStep = 5
    wIEsStart = 0
    wIEsStop = 35
    wIEsStep = 5

    trueWEI = 50.0
    trueWIE = 15.0

    indexWEIs = 0
    indexWIEs = 1
    indexDWEIs = 2
    indexDWIEs = 3
    indexLLs = 4

    minWeiArrows = 40
    maxWeiArrows = 60
    minWieArrows = 5
    maxWieArrows = 25

    estimationPathColor = "Green"
    estimationPathLS = "-"
    estimationPathLW = 1
    estimationPathMarker = "+"

    estimationResultFilename = "results/twoPopulationsSimpleMLEstimatesT0%.02fTf%.02fDt%fTrueWEI%.02fTrueWIE%.02fWEI0%.02fWIE0%.02fStepSize%fTol%fMaxIter%dYsSigma%.02f.npz" % (t0, tf, dt, trueWEI, trueWIE, wEI0, wIE0, stepSize, tol, maxIter, ysSigma)
    llsAndGradientsFilenamePattern = "%s/twoPopulationsSimpleLLsAndGradientsT0%.02fTf%.02fDt%fTrueWEI%.02fTrueWIE%.02fWEI%.02f-%.02f-%.02fWIE%.02f-%.02f-%.02fSigma%.02f.%s"
    llsAndGradientsFilename = llsAndGradientsFilenamePattern % ("results", t0, tf, dt, trueWEI, trueWIE, wEIsStart, wEIsStop, wEIsStep, wIEsStart, wIEsStop, wIEsStep, ysSigma, "npz")
    llsAndGradientsFigFilename = llsAndGradientsFilenamePattern % ("figures", t0, tf, dt, trueWEI, trueWIE, wEIsStart, wEIsStop, wEIsStep, wIEsStart, wIEsStop, wIEsStep, ysSigma, "gradient.eps")

    loadRes = np.load(llsAndGradientsFilename)
    llsAndGradients = loadRes["logLLsAndGradients"]

    loadRes = np.load(estimationResultFilename)
    ws = loadRes["ws"]

    weis = np.sort(np.unique(llsAndGradients[:, indexWEIs]))
    wies = np.sort(np.unique(llsAndGradients[:, indexWIEs]))
    weisArrows = weis[np.logical_and(minWeiArrows<=weis, weis<=maxWeiArrows)]
    wiesArrows = wies[np.logical_and(minWieArrows<=wies, wies<=maxWieArrows)]
    X, Y = np.meshgrid(wies, weis)
    XArrows, YArrows = np.meshgrid(wiesArrows, weisArrows)
    arrowsX = np.empty(XArrows.shape)
    arrowsY = np.empty(XArrows.shape)
    lls = np.zeros(X.shape)

    for i in xrange(llsAndGradients.shape[0]):
        wei = llsAndGradients[i, indexWEIs]
        wie = llsAndGradients[i, indexWIEs]
        dWei = llsAndGradients[i, indexDWEIs]
        dWie = llsAndGradients[i, indexDWIEs]
        ll = llsAndGradients[i, indexLLs]

        indexWei = np.where(weis==wei)
        indexWie = np.where(wies==wie)
        if minWeiArrows<=wei and wei<=maxWeiArrows and \
           minWieArrows<=wie and wie<=maxWieArrows:
            indexWeiArrows = np.where(weisArrows==wei)
            indexWieArrows = np.where(wiesArrows==wie)
            arrowsX[indexWeiArrows, indexWieArrows] = dWie
            arrowsY[indexWeiArrows, indexWieArrows] = dWei
        lls[indexWei, indexWie] = ll

    dWei = weis[1] - weis[0]
    dWie = wies[1] - wies[0]
    N = np.sqrt(arrowsX**2+arrowsY**2)
    maxN = N.max()
    arrowsX, arrowsY = arrowsX/maxN, arrowsY/maxN

    tooSmallLLIndices = np.where(lls<minLLToPlot)
    lls[tooSmallLLIndices[0], tooSmallLLIndices[1]] = minLLToPlot

    cs = plt.contourf(X, Y, lls, nLevels)
    plt.quiver(XArrows, YArrows, arrowsX, arrowsY, color=arrowsColor)
    cb = plt.colorbar(cs)
    cb.set_label("Log Likelihood", fontsize="large")

    # plt.ylim([weis[0]-dWei, weis[-1]+dWei])
    # plt.xlim([wies[0]-dWie, wies[-1]+dWie])

    plt.ylabel(r"$W_{ei}$", fontsize="large")
    plt.xlabel(r"$W_{ie}$")
    plt.title(r"$\sigma$=%.02f" % ysSigma)

    # plt.grid()
    plt.savefig(llsAndGradientsFigFilename)
    plt.close()
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

