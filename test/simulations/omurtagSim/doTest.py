
import sys
import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ifEDMsFunctions import computeQs, computeA0, computeA1, computeA2, computeQRs 

def main(argv):

    nVSteps = 5
    leakage = 1.0
    n0 = 6.0
    hMu = n0/210
    hSigma = hMu*0.3
    kappaMu = 0.03
    kappaSigma = kappaMu*0.3
    a0 = computeA0(nVSteps=nVSteps, leakage=leakage)
    # qs = computeQs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    a1 = computeA1(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    a2 = computeA2(nVSteps=nVSteps, kappaMu=kappaMu, kappaSigma=kappaSigma)
    # qrs = computeQRs(nVSteps=nVSteps, hMu=hMu, hSigma=hSigma)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

