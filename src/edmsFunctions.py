
import pdb
import numpy as np
import matplotlib.pyplot as plt
from myInformationTheory import computeKLDistance

def computeKLDistances(rhos1, rhos2):
    klDistances = np.empty(rhos1.shape[1])
    for j in xrange(klDistances.size):
        klDistances[j] = computeKLDistance(p=rhos1[:, j], q=rhos2[:, j])
    return(klDistances)

