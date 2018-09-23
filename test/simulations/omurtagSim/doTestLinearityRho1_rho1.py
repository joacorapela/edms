
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from plotEDMsResults import plotRhos

def main(argv):
    rho0ResultsFilename = 'results/testLinearityRho1_rho0.npz'
    rhoResultsFilename = 'results/testLinearityRho1_rho.npz'
    rho1ResultsFilename = 'results/testLinearityRho1_rho1.npz'
    rhosRho1FigsFilename  = 'figures/rhosTestLinearityRho1_rho1.eps'
    rhosKey = "rhos"
    vsKey = "vs"
    tsKey = "ts"
    fromTime = .01
    toTime = .95

    rho0Results = np.load(rho0ResultsFilename)
    rhoResults = np.load(rhoResultsFilename)
    vs = rho0Results[vsKey]
    ts = rho0Results[tsKey]
    rho0Rhos = rho0Results[rhosKey]
    rhoRhos = rhoResults[rhosKey]
    rho1Rhos = rhoRhos-rho0Rhos
    np.savez(rho1ResultsFilename, ts=ts, vs=vs, rhos=rho1Rhos)

    fromIndex = np.argmax(ts>=fromTime)
    toIndex = np.argmax(ts>toTime)
    plotRhos(vs, ts[fromIndex:toIndex], rho1Rhos[:, fromIndex:toIndex])
    plt.savefig(rhosRho1FigsFilename)

if __name__ == "__main__":
    main(sys.argv)

