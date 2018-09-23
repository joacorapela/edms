
import sys
import pdb
import matplotlib.pyplot as plt
import pickle
from ifEDMsFunctions import computeA0, computeA1, computeA2
from EigenReposForTwoStim import EigenReposForTwoStim

def main(argv):
    eigenReposFilenamePattern = \
     'results/anEigenResposForTwoStimNVSteps%dLeakage%fHMu%fHSigma%fEStimFrom%fEStimTo%fEStimStep%fIStimFrom%fIStimTo%fIStimStep%fNEigen%d_new.pickle'
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3

    startStimE = 1000
    minStimE = 0.0
    maxStimE = 2000.0
#     minStimE = 20.0
#     maxStimE = 100.0
    stepStimE = 20.0

    startStimI = 160
    minStimI = 0.0
    maxStimI = 300.0
#     minStimI = 0.0
#     maxStimI = 80.0
    stepStimI = 20.0

    nEigen = 17

    a0 = computeA0(nVSteps=nVSteps, leakage=leakage)
    a1 = computeA1(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu, kappaSigma=kappaSigma)
    eigenRepos = EigenReposForTwoStim()
    eigenRepos.computeEigenDecompositions(a0=a0, a1=a1, a2=a2,
                                                 startStimE=startStimE,
                                                 minStimE=minStimE, 
                                                 maxStimE=maxStimE,
                                                 stepStimE=stepStimE,
                                                 startStimI=startStimI,
                                                 minStimI=minStimI, 
                                                 maxStimI=maxStimI,
                                                 stepStimI=stepStimI,
                                                 nEigen=nEigen)
    eigenReposFilename = eigenReposFilenamePattern % \
                          (nVSteps, leakage, hMu, hSigma, 
                                    minStimE, maxStimE, stepStimE,
                                    minStimI, maxStimI, stepStimI,
                                    nEigen)
    with open(eigenReposFilename, 'wb') as f:
        pickle.dump(eigenRepos, f)

if __name__ == "__main__":
    main(sys.argv)
