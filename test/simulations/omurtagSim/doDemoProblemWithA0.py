
import sys
import numpy as np
import matplotlib.pyplot as plt
from plotEDMsResults import plotRhos
import pdb

def main(argv):
    t0 = 0
    tf = 0.25
    dt = 1e-5
    nTSteps = int(round((tf-t0)/dt))
    nVSteps = 210
    leakage = 20

    dv = 1.0/nVSteps
    mu = 0.50
    sigma2 = (0.1)**2
    newVs = (np.linspace(0, nVSteps, nVSteps))/nVSteps
    omurtagEtAlVs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps
    newRho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(newVs-mu)**2/(2*sigma2))
    omurtagEtAlRho0 = 1.0/np.sqrt(2*np.pi*sigma2)*np.exp(-(omurtagEtAlVs)**2/(2*sigma2))

    newA0 = computeNewA0(nVSteps=nVSteps, leakage=leakage)
    newEigRes = np.linalg.eig(newA0)
    omurtagEtAlA0 = computeOmurtagEtAlA0(nVSteps=nVSteps, leakage=leakage)
    omurtagEtAlEigRes = np.linalg.eig(omurtagEtAlA0)

    newRho = newRho0
    omurtagEtAlRho = omurtagEtAlRho0
    newRhos = np.empty((len(newRho0), nTSteps))
    omurtagEtAlRhos = np.empty((len(omurtagEtAlRho0), nTSteps))
    for i in xrange(nTSteps):
        newRhoDot = newA0.dot(newRho)
        newRho = newRho+dt*newRhoDot
        newRhos[:, i] = newRho

        omurtagEtAlRhoDot = omurtagEtAlA0.dot(omurtagEtAlRho)
        omurtagEtAlRho = omurtagEtAlRho+dt*omurtagEtAlRhoDot
        omurtagEtAlRhos[:, i] = omurtagEtAlRho

    ts = np.arange(start=t0, stop=tf+dt, step=dt)

    plt.close('all')

    plt.figure()
    sortedEValsIndices = np.argsort(abs(newEigRes[0]))
    plt.plot(newVs, newEigRes[1][:, sortedEValsIndices[0]])
    plt.xlabel('Voltage')
    plt.savefig('figures/a0Problem/goodEigVec0.eps')

    plt.figure()
    sortedEValsIndices = np.argsort(abs(omurtagEtAlEigRes[0]))
    plt.plot(omurtagEtAlVs, omurtagEtAlEigRes[1][:, sortedEValsIndices[0]])
    plt.xlabel('Voltage')
    plt.savefig('figures/a0Problem/poorEigVec0.eps')

    plt.figure()
    plotRhos(vs=newVs, times=ts, rhos=newRhos)
    plt.savefig('figures/a0Problem/goodRhos.eps')

    plt.figure()
    plotRhos(vs=omurtagEtAlVs, times=ts, rhos=omurtagEtAlRhos)
    plt.savefig('figures/a0Problem/poorRhos.eps')

def computeNewA0(nVSteps, leakage):
    a0 = np.diag(-np.arange(start=0, stop=nVSteps)) + \
         np.diag(np.arange(start=1, stop=nVSteps), k=1)
    return(a0*leakage)

def computeOmurtagEtAlA0(nVSteps, leakage):
    a0 = np.diag(np.ones(nVSteps))
    a0[nVSteps-1, nVSteps-1] = -(nVSteps-1)
    aSeq = np.arange(1, nVSteps)
    a0 = a0 + np.diag(aSeq, 1)
    a0 = a0 + np.diag(-aSeq, -1)
    return(a0*leakage/2)

if __name__ == "__main__":
    main(sys.argv)

