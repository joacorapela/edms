
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

def main(argv):
    ysSigma = 0.1
    resultsFilename = "results/twoPopulationsMLGradientsT0%.02fTf%.02fDt%fWEI%sWIE%sSigma%.02f.npz" % (t0, tfSaveYs, dt, str(trueWEI), str(trueWIE), ysSigma)
    derivs = np.load(derivsFilename)['derivs']
    weis = np.sort(np.unique(derivs[:, 0]))
    wies = np.sort(np.unique(derivs[:, 1]))
    X, Y = np.meshgrid(wies, weis)
    U = np.zeros(X.shape)
    V = np.zeros(X.shape)

    for i in xrange(derivs.shape[0]):
        wei = derivs[i, 0]
        wie = derivs[i, 1]
        dWei = derivs[i, 2]
        dWie = derivs[i, 3]

        indexWei = np.where(weis==wei)
        indexWie = np.where(wies==wie)
        U[indexWei, indexWie] = dWei
        V[indexWei, indexWie] = dWie

    dWei = weis[1] - weis[0]
    dWie = wies[1] - wies[0]
    N = np.sqrt(U**2+V**2)
    maxN = N.max()
    U, V = U/maxN, V/maxN
    plt.quiver(X, Y, U, V, color="blue")
    plt.xlabel(r"$W_{ei}$", fontsize="large")
    plt.ylabel(r"$W_{ie}$")
    plt.ylim([weis[0]-dWei, weis[-1]+dWei])
    plt.xlim([wies[0]-dWie, wies[-1]+dWie])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

