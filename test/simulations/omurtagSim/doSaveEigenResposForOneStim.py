
import sys
import pdb
import pickle
from IFEnsembleDensityIntegratorRNFNI import IFEnsembleDensityIntegratorRNFNI
from EigenReposForOneStim import EigenReposForOneStim

def main(argv):
    eigenReposFilenamePattern = \
     'results/anEigenResposForOneStimNVSteps%dLeakage%fHMu%fHSigma%fStimFrom%fStimTo%fStimStep%fNEigen%d.pickle'
    nVSteps = 210
    leakage = 20
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3

    minStim = 300
    maxStim = 1300
    stepStim = 20.0
    nEigen = 17

    ifEDIntegrator = \
     IFEnsembleDensityIntegratorRNFNI(leakage=leakage, hMu=hMu, hSigma=hSigma, 
                                                       nVSteps=nVSteps)
    a0 = ifEDIntegrator._a0
    a1 = ifEDIntegrator._a1
    eigenRepos = EigenReposForOneStim()
    eigenRepos.computeEigenDecompositions(a0=a0, a1=a1, 
                                                 minStim=minStim, 
                                                 maxStim=maxStim,
                                                 stepStim=stepStim,
                                                 nEigen=nEigen)
    eigenReposFilename = eigenReposFilenamePattern % \
                          (nVSteps, leakage, hMu, hSigma, minStim, maxStim, 
                                    stepStim, nEigen)
    with open(eigenReposFilename, 'wb') as f:
        pickle.dump(eigenRepos, f)

if __name__ == "__main__":
    main(sys.argv)
