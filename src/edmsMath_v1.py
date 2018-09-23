
import numpy as np
import math

def multiplyMatrixColsByScalars(m, s):
    return(m*s.reshape(1, len(s)))

def normalizedCrossCorrelation(x, y):
    answer = np.correlate(x-np.mean(x), y-np.mean(y))[0]
    answer = answer/(np.std(x)*np.std(y))
    return(answer)

def calculateLLs(ys, rs, ysSigma):
    lls = np.array(ys.size)
    auxSum = 0.0
    for i in xrange(lls.size):
        auxSum = auxSum + (ys[i]-rs[i])**2
        lls[i] = -0.5*((i+1)*log2PiYsSigma2+auxSum/ysSigma2)
    return(lls)
    return(-ys.size/2*math.log(2*math.pi*ysSigma**2)-\
            ((ys-rs)**2).sum()/(2*ysSigma**2))
