
import sys
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
from plotEDMsResults import plotKLDistancesFigure, plotSpikeRatesFigure,\
                            plotRhosAtT0Figure

def main(argv):

    populationType = "e"
    neuronsResultFilename = "results/twoPSinusoidalRWFWINNeurons9000.npz"
    fullEDMResulsFilename = "results/twoPSinusoidalRWFWIPopulation.npz"
    ldEDMREsultsFilenames = \
     ["results/twoPSinusoidalRWFWIPopulationNEigen17.npz", 
      "results/twoPSinusoidalRWFWIPopulationNEigen05.npz",
      "results/twoPSinusoidalRWFWIPopulationNEigen01.npz" 
      ]
    eigenReposFilename = "results/anEigenResposForTwoStimNVSteps210Leakage20.000000HMu0.028571HSigma0.008571EStimFrom0.000000EStimTo4000.000000EStimStep20.000000IStimFrom0.000000IStimTo600.000000IStimStep20.000000NEigen17.pickle"
    labels = ["Direct Sim.", 
              "EDM (full)", 
              "        (17 mb)", 
              "        (05 mb)",
              "        (01 mb)", 
              ]
    labelEInputCurrent = "     Exc. Input"
    legendLoc = "upper left"
    ylabelEInputCurrent = "Current (ips)"
    legendLocEInputCurrent = "upper left"
    bbox_to_anchor = (0.375, 0.975)
    bbox_to_anchorEInputCurrent = (0.375, 0.375)
    legendSize = 12
    colors = ["Grey", "Red", "Blue", "Cyan", "YellowGreen"]
    colEInputCurrent = "magenta"
    linestyles = ["-", "-", "-", "-", "-"]
    lsEInputCurrent = "--"
    linewidths = [1.0, 2.0, 2.0, 2.0, 2.0]

    dt = 1e-5
    spikesAverageWinTimeLength = 1e-3
    rhosAverageWinTimeLength = 5e-3
    xlab="Time (sec)"
    fromTime = 0.60
    toTime = 1.19
    xlab = "Time (sec)"
    title = ""
    sRsAndKLsPlotFilename = \
     "figures/%sPopSpikeRatesAndKLDistances170501.eps" % \
     populationType


    ylab = "Firing Rate (ips)"
    ylim= (0, 33)
    f, axarr = plt.subplots(2, sharex=True)
    plotSpikeRatesFigure(populationType=populationType,
                          neuronsResultFilename=neuronsResultFilename,
                          fullEDMResulsFilename=fullEDMResulsFilename,
                          ldEDMREsultsFilenames=ldEDMREsultsFilenames,
                          labels=labels, 
                          legendLoc=legendLoc,
                          bbox_to_anchor=bbox_to_anchor,
                          legendSize=legendSize,
                          colors=colors, linestyles=linestyles,
                          linewidths=linewidths, 
                          colEInputCurrent=colEInputCurrent,
                          lsEInputCurrent=lsEInputCurrent,
                          ylabelEInputCurrent=ylabelEInputCurrent,
                          labelEInputCurrent=labelEInputCurrent,
                          legendLocEInputCurrent=legendLocEInputCurrent,
                          bbox_to_anchorEInputCurrent=
                           bbox_to_anchorEInputCurrent,
                          dt=dt, 
                          averageWinTimeLength=spikesAverageWinTimeLength,
                          fromTime=fromTime, toTime=toTime,
                          xlab="", ylab=ylab, title=title, ylim=ylim,
                          ax=axarr[0])

    with open(eigenReposFilename, "rb") as f:
        eigenRepos = pickle.load(f)
#         eigenRepos = None
    ylab = "KL( Direct Sim. || EDM )"
    if populationType=="e":
        ylim = (0.00, 0.45)
    else:
        ylim = (0.00, 35.00)
    plotKLDistancesFigure(populationType=populationType,
                           neuronsResultFilename=neuronsResultFilename,
                           fullEDMResulsFilename=fullEDMResulsFilename,
                           ldEDMREsultsFilenames=ldEDMREsultsFilenames,
                           eigenRepos=eigenRepos,
                           labels=None, 
                           legendLoc=None,
                           bbox_to_anchor=None,
                           legendSize=None,
                           colors=colors[1:], 
                           linestyles=linestyles[1:],
                           linewidths=linewidths, dt=dt, 
                           averageWinTimeLength=rhosAverageWinTimeLength,
                           fromTime=fromTime, toTime=toTime,
                           xlab=xlab, ylab=ylab, title=title, ylim=ylim,
                           ax=axarr[1])
    plt.savefig(sRsAndKLsPlotFilename)

    xlab="$\upsilon$"
    title=""
    legendLoc="upper rigth"
    xlabFontsize = 20
    ylabFontsize = 20
    t0s = np.arange(start=0.60, stop=1.2, step=0.01)

    plt.close('all')
    for t0 in t0s:
        ylab=r"$\rho(\upsilon,\ %.02fs)$" % t0
        rhosAtT0PlotFilename="figures/%sPopulationRhosAt%.02f.eps" % \
                             (populationType, t0)
        plotRhosAtT0Figure(t0=t0, populationType=populationType,
                              neuronsResultFilename=neuronsResultFilename,
                              fullEDMResulsFilename=fullEDMResulsFilename,
                              ldEDMREsultsFilenames=ldEDMREsultsFilenames,
                              eigenRepos=eigenRepos,
                              labels=labels, 
                              legendLoc=legendLoc,
                              bbox_to_anchor=None, 
                              legendSize=None,
                              colors=colors, 
                              linestyles=linestyles,
                              linewidths=linewidths, 
                              xlab=xlab, ylab=ylab, title=title,
                              xlabFontsize=xlabFontsize,
                              ylabFontsize=ylabFontsize)
        plt.savefig(rhosAtT0PlotFilename)
        plt.close()

if __name__ == "__main__":
    main(sys.argv)

