
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

from myUtils import plotMultipleTimeSeries

   
def main(argv):
    dt = 1e-5
    averageWinTimeLength = 1e-3

    # firingRateSinusoidal
    neuronsResultsFN = "results/sinusoidalRNFNINNeurons9000.npz"
    populationResultsFN = "results/sinusoidalRNFNIPopulation.npz"
    firingRateFigureFN = "figures/firingRateSinusoidalRNFNI.eps"

    neuronsResults = np.load(neuronsResultsFN)
    populationResults = np.load(populationResultsFN)

    tsNeurons = neuronsResults["ts"]
    spikeRatesNeurons = neuronsResults["spikeRates"]
    tsPopulation = populationResults["ts"]
    spikeRatesPopulation = populationResults["spikeRates"]
    minNSamples = min(spikeRatesNeurons.size, spikeRatesPopulation.size)
    timesCol = np.column_stack((tsPopulation, tsNeurons))
    spikeRatesCol = np.column_stack((spikeRatesPopulation[:minNSamples], 
                                   spikeRatesNeurons[:minNSamples]))
    labels = ["EDM", "Direct simulation"]
    colors = ["red", "blue"]
    linestyles = ["-", "-"]
    linewidths = [1, 1]
    plotMultipleTimeSeries(timesCol=timesCol, 
                            timeSeriesCol=spikeRatesCol, 
                            colors=colors,
                            linestyles=linestyles,
                            linewidths=linewidths,
                            dt=tsNeurons[1]-tsNeurons[0], 
                            averageWinTimeLength=averageWinTimeLength, 
                            ylab="Firing Rate (ips)",
                            labels=labels, 
                            fromTime=0.7, toTime=0.95)
    plt.grid()
    plt.savefig(firingRateFigureFN)
    #

    # firingRateStep and rhoStep
#     neuronsResultsFN = "results/stepRNFNINNeurons9000.npz"
#     populationResultsFN = "results/stepRNFNIPopulation.npz"
#     firingRateFigureFN = "figures/firingRateStepRNFNI.eps"
#     rhosFigureFN = "figures/rhosAtSSStepRNFNI.eps"

#     neuronsResults = np.load(neuronsResultsFN)
#     populationResults = np.load(populationResultsFN)

#     ts = neuronsResults["ts"]
#     spikeRatesNeurons = neuronsResults["spikeRates"]
#     spikeRatesPopulation = populationResults["spikeRates"]
#     minNSamples = min(spikeRatesNeurons.size, spikeRatesPopulation.size)
#     spikeRates = np.column_stack((spikeRatesPopulation[:minNSamples], 
#                                    spikeRatesNeurons[:minNSamples]))
#     labels = ["EDM", "Direct simulation"]
#     colors = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
#     linestyles = ["-", "-"]
#     plotMultipleSpikeRates(ts=ts, spikeRates=spikeRates, labels=labels,
#                                dt=ts[1]-ts[0], colors=colors,
#                                linestyles=linestyles,
#                                averageWinTimeLength=averageWinTimeLength, 
#                                plotFilename=firingRateFigureFN,
#                                fromTime=0.0, toTime=0.4)

#     ssTime = 0.4
#     ssIndex = np.argmax(ts>ssTime)
#     rhosAtSSNeurons = neuronsResults["rhos"][:,ssIndex]
#     rhosAtSSPopulation = populationResults["rhos"][:,ssIndex]
#     nVSteps = neuronsResults["rhos"].shape[0]
#     vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
#     plotMultipleRhosAtT0(vs, rhosAtSSNeurons, rhosAtSSPopulation, 
#                              rhosFigureFN)
    #

    # noiseless firingRateStep
#     neuronsResultsFN = "results/stepNoiselessNNeurons90000.npz"
#     populationResultsFN = "results/stepNoiselessPopulation.npz"
#     firingRateFigureFN = "figures/noiselessFiringRateStep.eps"
#     rhosFigureFN = "figures/noiselessRhoAtSSStep.eps"
# 
#     neuronsResults = np.load(neuronsResultsFN)
#     populationResults = np.load(populationResultsFN)
# 
#     ts = neuronsResults["ts"]
#     spikeRatesNeurons = neuronsResults["spikeRates"]
#     spikeRatesPopulation = populationResults["spikeRates"]
#     minNSamples = min(spikeRatesNeurons.size, spikeRatesPopulation.size)
#     spikeRates = np.column_stack((spikeRatesPopulation[:minNSamples],
#                                    spikeRatesNeurons[:minNSamples]))
#     labels = ["EDM", "Direct simulation"]
#     colors = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
#     linestyles = ["-", "-"]
#     plotMultipleSpikeRates(ts, spikeRates, labels=labels,
#                                dt=ts[1]-ts[0], colors=colors,
#                                linestyles=linestyles, 
#                                averageWinTimeLength=averageWinTimeLength, 
#                                plotFilename=firingRateFigureFN,
#                                fromTime=0, toTime=0.4)
# 
#     ssTime = 0.4
#     ssIndex = np.argmax(ts>ssTime)
#     rhosAtSSNeurons = neuronsResults["rhos"][:,ssIndex]
#     rhosAtSSPopulation = populationResults["rhos"][:,ssIndex]
#     nVSteps = neuronsResults["rhos"].shape[0]
#     vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
#     plotMultipleRhosAtT0(vs, rhosAtSSNeurons, rhosAtSSPopulation, 
#                              rhosFigureFN)
    # 

    # WFWI
#     neuronsWFNIResultsFN = "results/sinusoidalRWFNINNeurons9000.npz"
#     neuronsResultsFN = "results/sinusoidalRWFWINNeurons9000KMu0.0300.npz"
#     populationResultsFN = "results/sinusoidalRWFWIPopulationKMu0.0300.npz"
#     firingRateFigureFN = "figures/sinusoidalRWFWIFiringRate.eps"
#     rhosFigureFN = "figures/sinusoidalRWFWIRhos.eps"
# 
#     neuronsResults = np.load(neuronsResultsFN)
#     populationResults = np.load(populationResultsFN)
#     neuronsWFNIResults = np.load(neuronsWFNIResultsFN)
# 
#     ts = neuronsResults["ts"]
#     spikeRatesNeurons = neuronsResults["spikeRates"]
#     spikeRatesPopulation = populationResults["spikeRates"]
#     spikeRatesPopulationWFNI = neuronsWFNIResults["spikeRates"]
#     minNSamples = min(spikeRatesNeurons.size, spikeRatesPopulation.size,
#                                               spikeRatesPopulationWFNI.size)
#     spikeRates = np.column_stack((spikeRatesPopulation[:minNSamples], 
#                                    spikeRatesNeurons[:minNSamples], 
#                                    spikeRatesPopulationWFNI[:minNSamples]))
#     labels = ["EDM", "Direct simulation", "Direct simulation (no inhibition)"]
#     colors = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)]
#     linestyles = ["-", "-", "--"]
#     plotMultipleSpikeRates(ts=ts, spikeRates=spikeRates, labels=labels,
#                                dt=ts[1]-ts[0], colors=colors,
#                                linestyles=linestyles,
#                                averageWinTimeLength=averageWinTimeLength, 
#                                plotFilename=firingRateFigureFN,
#                                fromTime=0.7, toTime=0.97)
#     ssTime = 0.99
#     ssIndex = np.argmax(ts>ssTime)
#     rhosAtSSNeurons = neuronsResults["rhos"][:,ssIndex]
#     rhosAtSSPopulation = populationResults["rhos"][:,ssIndex]
#     nVSteps = neuronsResults["rhos"].shape[0]
#     vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
#     plotMultipleRhosAtT0(vs, rhosAtSSNeurons, rhosAtSSPopulation, 
#                              rhosFigureFN)
    # 

    # WFNI
#     neuronsResultsFN = "results/sinusoidalRWFNINNeurons9000.npz"
#     populationResultsFN = "results/sinusoidalRWFNIPopulation.npz"
#     firingRateFigureFN = "figures/sinusoidalRWFNIFiringRate.eps"
#     rhosFigureFN = "figures/sinusoidalRWFNIRhos.eps"
# 
#     neuronsResults = np.load(neuronsResultsFN)
#     populationResults = np.load(populationResultsFN)
# 
#     ts = neuronsResults["ts"]
#     spikeRatesNeurons = neuronsResults["spikeRates"]
#     spikeRatesPopulation = populationResults["spikeRates"]
#     minNSamples = min(spikeRatesNeurons.size, spikeRatesPopulation.size)
#     spikeRates = np.column_stack((spikeRatesPopulation[:minNSamples],
#                                    spikeRatesNeurons[:minNSamples]))
#     labels = ["EDM", "Direct simulation"]
#     colors = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]
#     linestyles = ["-", "-"]
#     plotMultipleSpikeRates(ts, spikeRates, labels=labels,
#                                dt=ts[1]-ts[0], colors=colors,
#                                linestyles=linestyles, 
#                                averageWinTimeLength=averageWinTimeLength, 
#                                plotFilename=firingRateFigureFN,
#                                fromTime=0.7, toTime=0.97)

#     ssTime = 0.99
#     ssIndex = np.argmax(ts>ssTime)
#     rhosAtSSNeurons = neuronsResults["rhos"][:,ssIndex]
#     rhosAtSSPopulation = populationResults["rhos"][:,ssIndex]
#     nVSteps = neuronsResults["rhos"].shape[0]
#     vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
#     plotMultipleRhosAtT0(vs, rhosAtSSNeurons, rhosAtSSPopulation, 
#                              rhosFigureFN)
    # 

if __name__ == "__main__":
    main(sys.argv)

