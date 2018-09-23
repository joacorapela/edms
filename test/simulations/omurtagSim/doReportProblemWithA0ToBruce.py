
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI

def main(argv):

#     a0 = np.array(([0, 1, 1, 0], [0, -1, 1, 2], [0, 0, -2, -2], [0, 0, 0, 0]))
#     eigRes = np.linalg.eig(a0)
#     pdb.set_trace()

#     a0 = np.array(([3, 0, 0, -9], [-3, 5, 0, 0], [0, -5, 7, 0], [0, 0, -7, 9]))
#     eigRes = np.linalg.eig(a0)
#     pdb.set_trace()

#     a0 = np.array(([2, 2, 0, 0], [-2, 1, 3, 0], [0, -3, 1, 4], [0, 0, -4, -4]))
#     eigRes = np.linalg.eig(a0)
#     pdb.set_trace()

#     a0 = np.array(([1, 1, 0, 0], [-1, 1, 2, 0], [0, -2, 1, 3], [0, 0, -3, -3]))
#     eigRes = np.linalg.eig(a0)
#     pdb.set_trace()

    nVSteps = 500
    leakage = 2
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    sEs = np.array([0, 1, 2, 4, 6, 8, 10])

    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRNFNI(nVSteps=nVSteps, leakage=leakage,
                                                       hMu=hMu, hSigma=hSigma)
    a0 = ifEDIntegrator._a0
    a1 = ifEDIntegrator._a1

    for i in xrange(sEs.size):
        sE = sEs[i]
        eigRes = np.linalg.eig(a0+sE*a1)
        sortedEValsIndices = np.argsort(abs(eigRes[0]))
        plt.plot(eigRes[1][:, sortedEValsIndices[0]], 
                              label=r"$\sigma_E$= %d, $\lambda$=%f+j%f" % \
                              (sE, eigRes[0][sortedEValsIndices[0]].real, 
                                eigRes[0][sortedEValsIndices[0]].imag))
    plt.legend();
    plt.show()
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

