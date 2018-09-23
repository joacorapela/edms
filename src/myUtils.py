
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt

def splitRealAndImaginaryPartsInVector(v):
    riV = np.empty(2*v.size)
    riV[::2] = v.real
    riV[1::2] = v.imag
    return(riV)

def splitRealAndImaginaryPartsInMatrix(m):
    riM = np.empty(tuple([z * 2 for z in m.shape]))

    for i in xrange(m.shape[0]):
        for j in xrange(m.shape[1]):
            riM[i*2, j*2] = m[i, j].real
            riM[i*2, j*2+1] = -m[i, j].imag
            riM[i*2+1, j*2] = m[i, j].imag
            riM[i*2+1, j*2+1] = m[i, j].real
    return(riM)

def getRealPartOfCArrayDotSRIVector(cArray, sriVector):
        return(cArray.real.dot(sriVector[::2])-cArray.imag.dot(sriVector[1::2]))

def averageVector(v, averageWinSampleLength):
    averagedV = np.zeros(len(v)/averageWinSampleLength)
    for i in xrange(len(averagedV)):
        minIndex = i*averageWinSampleLength
        maxIndex = (i+1)*averageWinSampleLength
        averagedV[i] = v[minIndex:maxIndex].mean()
    return(averagedV)

def plotTimeSeries(times, timeSeries, dt, averageWinTimeLength, ylab, 
                          label=None, 
                          xlab="Time (s)",
                          title="",
                          color="b",
                          linestyle="-",
                          linewidth=1.0,
                          ylim=None,
                          xlim=None,
                          ax=None):
    averageWinSampleLength = averageWinTimeLength/dt
    averagedTimeSeries = averageVector(timeSeries, averageWinSampleLength)
    averagedTimes = averageVector(times, averageWinSampleLength)
    if ax is not None:
        ax.plot(averagedTimes, averagedTimeSeries, color=color, 
                               linestyle=linestyle, linewidth=linewidth, 
                               label=label)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)
        if title is not None:
            ax.set_title(title)
    else:
        plt.plot(averagedTimes, averagedTimeSeries, color=color, 
                                linestyle=linestyle, linewidth=linewidth, 
                                label=label)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlab is not None:
            plt.xlabel(xlab)
        if ylab is not None:
            plt.ylabel(ylab)
        if title is not None:
            plt.title(title)

def plotMultipleTimeSeries(timesCol, timeSeriesCol, colors, 
                                     linestyles, linewidths, dt, 
                                     averageWinTimeLength, 
                                     ylab,
                                     labels=None, legendLoc=None,
                                     bbox_to_anchor=None,
                                     legendSize=None,
                                     fromTime=0.70, toTime=0.95, 
                                     xlab="Time (sec)", 
                                     title="", ylim=None, ax=None):
    def plotLegend(ax, legendLoc, bbox_to_anchor=None, legendSize=None):
        if ax is not None:
            if legendSize is not None:
                if bbox_to_anchor is not None:
                    ax.legend(loc=legendLoc, bbox_to_anchor=bbox_to_anchor,
                                             prop={"size":legendSize})
                else:
                    ax.legend(loc=legendLoc, prop={"size":legendSize})
            else:
                if bbox_to_anchor is not None:
                    ax.legend(loc=legendLoc, bbox_to_anchor=bbox_to_anchor)
                else:
                    ax.legend(loc=legendLoc)
        else:
            if legendSize is not None:
                if bbox_to_anchor is not None:
                    plt.legend(loc=legendLoc, bbox_to_anchor=bbox_to_anchor,
                                              prop={"size":legendSize})
                else:
                    plt.legend(loc=legendLoc, prop={"size":legendSize})
            else:
                if bbox_to_anchor is not None:
                    plt.legend(loc=legendLoc, bbox_to_anchor=bbox_to_anchor)
                else:
                    plt.legend(loc=legendLoc)
            plt.grid()

    xlim = (fromTime, toTime)
    for j in xrange(timeSeriesCol.shape[1]):
        fromIndex = np.argmax(timesCol[:, j]>=fromTime)
        toIndex = np.argmax(timesCol[:, j]>toTime)
        if labels is not None:
            label = labels[j]
        else:
            label = ""
        plotTimeSeries(times=timesCol[fromIndex:toIndex, j], 
                        timeSeries=timeSeriesCol[fromIndex:toIndex, j], 
                        dt=dt, 
                        averageWinTimeLength=averageWinTimeLength, 
                        xlab=xlab, 
                        ylab=ylab, 
                        title=title, 
                        label=label,
                        color=colors[j],
                        linestyle=linestyles[j],
                        linewidth=linewidths[j],
                        xlim=xlim,
                        ylim=ylim,
                        ax=ax)
    if labels is not None:
        plotLegend(ax=ax, legendLoc=legendLoc, bbox_to_anchor=bbox_to_anchor,
                          legendSize=legendSize)
    if ax is not None:
        ax.grid()
    else:
        plt.grid()
   
def generateJointParameters(parametersCol):
    def appendParameters(jointParameters, parameters):
        newJointParameters = np.zeros((jointParameters.shape[0]*parameters.size,
                                     jointParameters.shape[1]+1))
        for i in xrange(parameters.size):
            for j in xrange(jointParameters.shape[0]):
                newJointParameters[i*jointParameters.shape[0]+j,:] = \
                                 np.append(jointParameters[j,:], parameters[i])
        return(newJointParameters)

    jointParameters = np.reshape(parametersCol[0], (len(parametersCol[0]),1))
    for currentIndex in xrange(1, len(parametersCol)):
        jointParameters = appendParameters(jointParameters=jointParameters, 
                                      parameters=parametersCol[currentIndex])
    return(jointParameters)
