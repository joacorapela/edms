
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from plotEDMsResults import plotRhos

def main(argv):
    rho1ResultsFilename = 'results/testLinearityRho1_rho1.npz'
    linearRho1sFigsFilename = 'figures/linearRho1sTestLinearityRho1_finalTest.eps'
    diffRhosFigsFilename = 'figures/diffRhosTestLinearityRho1_finalTest.eps'
    rhosKey = "rhos"
    vsKey = "vs"
    tsKey = "ts"
    fromTime = .5
    toTime = .95
    tol = 1e-6

    sigma0 = 0
    b = 100
    freq = 4
    def sinusoidalInputMeanFrequency(t, sigma0=sigma0, b=b,
                                        omega=2*np.pi*freq):
        return(sigma0+b*np.sin(omega*t))

    rho1Results = np.load(rho1ResultsFilename)
    vs = rho1Results[vsKey]
    ts = rho1Results[tsKey]
    rho1s = rho1Results[rhosKey]

    fromIndex = np.argmax(ts>=fromTime)
    toIndex = np.argmax(ts>toTime)
    ts = ts[fromIndex:toIndex]
    rho1s = rho1s[:, fromIndex:toIndex]
    s1s = sinusoidalInputMeanFrequency(t=ts)
    indexZeroS1 = np.where(abs(s1s)<tol)[0][0]
    phi0 = rho1s[:, indexZeroS1]
    indexArgMaxS1 = np.argmax(abs(s1s))
    phi1 = (rho1s[:, indexArgMaxS1]-phi0)/s1s[indexArgMaxS1]
    linearRho1s = np.outer(phi1, s1s) + np.outer(phi0, np.ones(s1s.size))
    diffRhos = rho1s-linearRho1s

    plotRhos(vs, ts, linearRho1s)
    plt.savefig(linearRho1sFigsFilename)
    plt.close('all')

    plotRhos(vs, ts, diffRhos)
    plt.savefig(diffRhosFigsFilename)
    plt.close('all')

if __name__ == "__main__":
    main(sys.argv)

