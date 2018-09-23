
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pickle

def main(argv):
#     eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom0.000000EStimTo100.000000EStimStep20.000000IStimFrom0.000000IStimTo100.000000IStimStep20.000000NEigen17.pickle"
#     eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom760.000000EStimTo840.000000EStimStep20.000000IStimFrom160.000000IStimTo240.000000IStimStep20.000000NEigen17.pickle"
#     eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom20.000000EStimTo100.000000EStimStep20.000000IStimFrom0.000000IStimTo80.000000IStimStep20.000000NEigen17.pickle"
#     with open(eigenReposFilename, 'rb') as f:
#         eigenRepos = pickle.load(f)

    eigenRepos = EigenReposForTwoStim()
    
#     plt.close('all')

#     nEValsToPlot = 2
#     sE = eigenRepos._stimuliE[0]
#     sEIndex = np.argmin(abs(eigenRepos._stimuliE-sE))

#     plt.figure()
#     for eigenIndex in xrange(nEValsToPlot):
#         plt.plot(eigenRepos._stimuliI, eigenRepos._eVals[eigenIndex, sEIndex, :].real, label="EVal Index %d" % (eigenIndex))
#     plt.xlabel("Inhibitory input")
#     plt.ylabel("Eigenvalue")
#     plt.title("Eigenvalues for excitatory input = %.02f" % sE)
#     plt.legend()
#     plt.grid()
#     plt.show()
#     pdb.set_trace()

#     sE = eigenRepos._stimuliE[0]
# 
#     eVecToPlot = 0
#     plt.figure()
#     for iI in xrange(eigenRepos._stimuliI.size):
#         sI = eigenRepos._stimuliI[iI]
#         eVecs = eigenRepos.getEigenvectors(sE=sE, sI=sI)
#         plt.plot(eVecs[:, eVecToPlot].real, label="Inh. Stim. %.02f" % (sI))
#     plt.title("Eigenvectors for excitatory input = %.02f" % sE)
#     plt.legend()
#     plt.grid()
# 
#     sI = eigenRepos._stimuliI[0]
# 
#     eVecToPlot = 0
#     plt.figure()
#     fromIE = 0
#     toIE = 4
#     for iE in xrange(fromIE, toIE):
#         sE = eigenRepos._stimuliE[iE]
#         eVecs = eigenRepos.getEigenvectors(sE=sE, sI=sI)
#         plt.plot(eVecs[:, eVecToPlot].real, label="Exc. Stim. %.02f" % (sE))
#     plt.title("Eigenvectors for inhibitory input = %.02f" % sI)
#     plt.legend()
#     plt.grid()
#     plt.show()

#     plt.close("all")
#     sEs = [0]
#     sIs = [0, 10, 50, 100, 200, 310]
#     f, axarr = plt.subplots(2)
#     for iE in xrange(len(sEs)):
#         sE = sEs[iE]
#         for iI in xrange(len(sIs)):
#             sI = sIs[iI]
#             dEValsE, dEValsI = eigenRepos.getDEVals(sE=sE, sI=sI)
#             axarr[0].plot(dEValsE, label="(sE=%d, sI=%d)"%(sE, sI))
#             axarr[1].plot(dEValsI, label="(sE=%d, sI=%d)"%(sE, sI))
#     axarr[0].legend()
#     axarr[1].legend()

#     eVecToPlot = 0
#     f, axarr = plt.subplots(2)
#     for iE in xrange(len(sEs)):
#         sE = sEs[iE]
#         for iI in xrange(len(sIs)):
#             sI = sIs[iI]
#             dEVecsE, dEVecsI = eigenRepos.getDEVecs(sE=sE, sI=sI)
#             axarr[0].plot(dEVecsE[:, eVecToPlot], label="(sE=%d, sI=%d)"%(sE, sI))
#             axarr[1].plot(dEVecsI[:, eVecToPlot], label="(sE=%d, sI=%d)"%(sE, sI))
#     axarr[0].legend()
#     axarr[1].legend()

#     plt.show()

if __name__ == "__main__":
    main(sys.argv)

