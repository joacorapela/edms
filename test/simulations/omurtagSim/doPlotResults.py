
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from plotEDMsResults import plotSinglePopulationResults, plotTwoPopulationsResults, plotTwoPopulationsLDResults

def main(argv):
    resultsFilename = "results/stepDDiffusionNFNIPopulation.npz"
    spikeRatesFigFilename = \
     "figures/spikeRatesStepDDiffusionNFNIPopulation.eps"
    rhosFigFilename = "figures/rhosStepDDiffusionNFNIPopulation.eps"
    plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
                                                 rhosFigFilename,
#                                                  ylimAx1=(0, 50),
                                                 fromTime=0.01,
                                                 toTime=0.95)

#     resultsFilename = "results/sinusoidalRNFNINNeurons9000.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRNFNINNeurons9000.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRNFNINNeurons9000.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename,
#                                                  fromTime=0.70,
#                                                  toTime=0.95,
#                                                  ylimAx1=(0, 50),
#                                                  climMin=0.0,
#                                                  climMax=7.3)
# 
#     resultsFilename = "results/sinusoidalRNFNIPopulation.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRNFNIPopulation.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRNFNIPopulation.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename,
#                                                  fromTime = 0.7,
#                                                  toTime=0.95,
#                                                  climMin=0.0,
#                                                  climMax=7.3)

#     resultsFilename = "results/sinusoidalRNFNIPopulationB.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRNFNIPopulationB.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRNFNIPopulationB.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename,
#                                                  startTimePlotRhos = 0.0,
#                                                  endTimePlotRhos=1.00,
#                                                  ylimAx1=(0, 50))

#     resultsFilename = "results/zeroRNFNIPopulationB.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesZeroRNFNIPopulationB.eps"
#     rhosFigFilename = "figures/rhosZeroRNFNIPopulationB.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename,
#                                                  fromTime=0.0,
#                                                  toTime=0.99,
#                                                  ylimAx1=(0, 50),
#                                                  climMax=10,
#                                                  climMin=-10)

#     resultsFilename = "results/sinusoidalRNFWINNeurons9000KMu0.0300.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRNFWINNeurons9000KMu0.0300.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRNFWINNeurons9000KMu0.0300.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename,
#                                                  startTimePlotRhos = 0.7,
#                                                  endTimePlotRhos=0.95)

#     resultsFilename = "results/testLinearityRho1_rho0.npz"
#     t0 = 0.8
#     rhoAtT0FigFilename = "figures/rhoAt%.2fTestLinearityRho1_rho0.eps"%(t0)
#     rhosKey = "rhos"
#     timesKey = "ts"
#     vsKey = "vs"
#     xlab = "Voltage"
#     ylab = "Probability"
#     results = np.load(resultsFilename)
#     rhos = results[rhosKey]
#     times = results[timesKey]
#     t0Index = np.argmax(times>t0)
#     rhoAtT0 = rhos[:, t0Index]
#     vs = results[vsKey]
#     plt.plot(vs, rhoAtT0)
#     plt.xlabel(xlab)
#     plt.ylabel(ylab)
#     plt.grid()
#     plt.savefig(rhoAtT0FigFilename)
#     plt.close('all')
#     
#     resultsFilename = "results/testLinearityRho1_rho0.npz"
#     spikeRatesFigFilename = "figures/spikeRatesTestLinearityRho1_rho0.eps"
#     rhosFigFilename = "figures/rhosTestLinearityRho1_rho0.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                         rhosFigFilename, 
#                                         fromTime=0.01, toTime=0.95)
#     plt.close('all')

#     resultsFilename = "results/testLinearityRho1_rho.npz"
#     t0 = 0.8
#     rhoAtT0FigFilename = "figures/rhoAt%.2fTestLinearityRho1_rho.eps"%(t0)
#     rhosKey = "rhos"
#     timesKey = "ts"
#     vsKey = "vs"
#     xlab = "Voltage"
#     ylab = "Probability"
#     results = np.load(resultsFilename)
#     rhos = results[rhosKey]
#     times = results[timesKey]
#     t0Index = np.argmax(times>t0)
#     rhoAtT0 = rhos[:, t0Index]
#     vs = results[vsKey]
#     plt.plot(vs, rhoAtT0)
#     plt.xlabel(xlab)
#     plt.ylabel(ylab)
#     plt.grid()
#     plt.savefig(rhoAtT0FigFilename)
#     plt.close('all')
#     
#     resultsFilename = "results/testLinearityRho1_rho.npz"
#     spikeRatesFigFilename = "figures/spikeRatesTestLinearityRho1_rho.eps"
#     rhosFigFilename = "figures/rhosTestLinearityRho1_rho.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                         rhosFigFilename, 
#                                         fromTime=0.01, toTime=0.95)
#     plt.close('all')
# 
#     resultsFilename = "results/sinusoidalRWFWINNeurons9000KMu0.0300.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRWFWINNeurons9000KMu0.0300.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRWFWINNeurons9000KMu0.0300.eps"
#     plotOmurtagResults(resultsFilename, spikeRatesFigFilename, 
#                                         rhosFigFilename)

#     resultsFilename = "results/sinusoidalRWFNINNeurons900.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRWFNINNeurons900.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRWFNINNeurons900.eps"
#     plotOmurtagResults(resultsFilename, spikeRatesFigFilename, 
#                                         rhosFigFilename)

#     resultsFilename = "results/sinusoidalRWFWINNeurons9000KMu0.0300.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRWFWINNeurons9000KMu0.0300.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRWFWINNeurons9000KMu0.0300.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                         rhosFigFilename)

#     resultsFilename = "results/sinusoidalRWFWINNeurons900KMu0.0300.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRWFWINNeurons900KMu0.0300.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRWFWINNeurons900KMu0.0300.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename)

#     resultsFilename = "results/sinusoidalRNFWIPopulationKMu0.0300.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRNFWIPopulationKMu0.0300.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRNFWIPopulationKMu0.0300.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename)

#     resultsFilename = "results/sinusoidalRWFWIPopulationKMu0.0300.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalRWFWIPopulationKMu0.0300.eps"
#     rhosFigFilename = "figures/rhosSinusoidalRWFWIPopulationKMu0.0300.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename)

#     resultsFilename = "results/sinusoidalLDRWFWIPopulationNEigen17.npz"
#     spikeRatesFigFilename = \
#      "figures/spikeRatesSinusoidalLDRWFWIPopulationNEigen17.eps"
#     results = np.load(resultsFilename)
#     plotSpikeRates(ts=results["ts"], spikeRates=results["spikeRates"], dt=dt, 
#     rhosFigFilename = "figures/rhosSinusoidalRWFWIPopulationKMu0.0300.eps"
#     plotSinglePopulationResults(resultsFilename, spikeRatesFigFilename, 
#                                                  rhosFigFilename)

#     resultsFilename = "results/twoPSinusoidalRNFWI_RNFNIPopulation.npz"
#     edm1SpikeRatesFigFilename = \
#      "figures/twoPEDM1SpikeRatesSinusoidalRNFWI_RNFNIPopulation.eps"
#     edm2SpikeRatesFigFilename = \
#      "figures/twoPEDM2SpikeRatesSinusoidalRNFWI_RNFNIPopulation.eps"
#     plotTwoPopulationsResults(resultsFilename=resultsFilename, 
#                                edm1SpikeRatesFigFilename=
#                                 edm1SpikeRatesFigFilename, 
#                                edm2SpikeRatesFigFilename=
#                                 edm2SpikeRatesFigFilename,
#                                ylimAx1=c(0, 25),
#                                ylimAx2=c(0, 1400))

#     resultsFilename = "results/twoPSinusoidalRWFWI2NNeurons9000b.npz"
#     edm1SpikeRatesFigFilename = \
#      "figures/twoPEDM1SinusoidalRWFWI2NNeurons9000.eps"
#     edm2SpikeRatesFigFilename = \
#      "figures/twoPEDM2SinusoidalRWFWI2NNeurons9000.eps"
#     plotTwoPopulationsResults(resultsFilename, edm1SpikeRatesFigFilename, 
#                                                edm2SpikeRatesFigFilename,
#                                                fromTime=0.0, toTime=0.94)

#     resultsFilename = "results/twoPSinusoidalRWFWINNeurons9000.npz"
#     eSpikeRatesFigFilename = \
#      "figures/twoPESpikeRatesSinusoidalRWFWINNeurons9000.eps"
#     iSpikeRatesFigFilename = \
#      "figures/twoPISpikeRatesSinusoidalRWFWINNeurons9000.eps"
#     eRhosFigFilename = \
#      "figures/twoPERhosSinusoidalRWFWINNeurons9000.eps"
#     iRhosFigFilename = \
#      "figures/twoPIRhosSinusoidalRWFWINNeurons9000.eps"
#     plotTwoPopulationsResults(resultsFilename=resultsFilename, 
#                                eSpikeRatesFigFilename=eSpikeRatesFigFilename, 
#                                iSpikeRatesFigFilename=iSpikeRatesFigFilename,
#                                eRhosFigFilename=eRhosFigFilename,
#                                iRhosFigFilename=iRhosFigFilename,
#                                fromTime=0.6, toTime=1.19,
#                                climMin=0.0, climMax=5.00,
#                                ylimAx1=(0, 25),
#                                ylimAx2=(0, 1400))
# 
#     resultsFilename = "results/twoPSinusoidalRWFWIPopulation.npz"
#     eSpikeRatesFigFilename = \
#      "figures/twoPESpikeRatesSinusoidalRWFWIPopulation.eps"
#     iSpikeRatesFigFilename = \
#    "figures/twoPISpikeRatesSinusoidalRWFWIPopulation.eps"
#     eRhosFigFilename = \
#      "figures/twoPERhosSinusoidalRWFWIPopulation.eps"
#     iRhosFigFilename = \
#      "figures/twoPIRhosSinusoidalRWFWIPopulation.eps"
#     plotTwoPopulationsResults(resultsFilename=resultsFilename, 
#                                eSpikeRatesFigFilename=eSpikeRatesFigFilename, 
#                                iSpikeRatesFigFilename=iSpikeRatesFigFilename, 
#                                eRhosFigFilename=eRhosFigFilename,
#                                iRhosFigFilename=iRhosFigFilename,
#                                fromTime=0.6, toTime=1.19,
#                                climMin=0.0, climMax=5.00,
#                                ylimAx1=(0, 25),
#                                ylimAx2=(0, 1400))


#     nEigen = 10
#     resultsFilename = "results/twoPSinusoidalRWFWIPopulationNEigen%02d.npz" % nEigen
#     eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom0.000000EStimTo4000.000000EStimStep20.000000IStimFrom0.000000IStimTo600.000000IStimStep20.000000NEigen17.pickle"
#     eSpikeRatesFigFilename = \
#      "figures/twoPSinusoidalRWFWIPopulationNEigen%02dESpikeRates.eps" % nEigen
#     iSpikeRatesFigFilename = \
#      "figures/twoPSinusoidalRWFWIPopulationNEigen%02dISpikeRates.eps" % nEigen
#     eRhosFigFilename = \
#      "figures/twoPSinusoidalRWFWIPopulationNEigen%02dERhos.eps" % nEigen
#     iRhosFigFilename = \
#      "figures/twoPSinusoidalRWFWIPopulationNEigen%02dIRhos.eps" % nEigen
#     with open(eigenReposFilename, "rb") as f:
#         eigenRepos = pickle.load(f)
#     plotTwoPopulationsLDResults(resultsFilename=resultsFilename, 
#                                  eigenRepos=eigenRepos,
#                                  eSpikeRatesFigFilename=eSpikeRatesFigFilename, 
#                                  iSpikeRatesFigFilename=iSpikeRatesFigFilename,
#                                  eRhosFigFilename=eRhosFigFilename,
#                                  iRhosFigFilename=iRhosFigFilename,
#                                  fromTime=0.6, toTime=1.19)

if __name__ == "__main__":
    main(sys.argv)

