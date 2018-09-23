
import sys
import numpy as np

class TwoPopulationsIFEnsembleDensitySimpleIntegrator:

    def __init__(self, a0, a1, a2, dt, dv, reversedQs):
        self._a0 = a0
        self._a1 = a1
        self._a2 = a2
        self._dt =dt
        self._dv =dv
        self._reversedQs =reversedQs

    def integrate(self, eRho0, iRho0, wEI, wIE, eInputs):
        eRs = np.empty(eInputs.size)
        iRs = np.empty(eInputs.size)
        eRs[0] = self._dv*eInputs[0]*self._reversedQs.dot(eRho0)
        iRs[0] = 0
        eRhos = np.empty((self._a0.shape[0], eInputs.size))
        iRhos = np.empty((self._a0.shape[0], eInputs.size))
        eRhos[:, 0] = eRho0
        iRhos[:, 0] = iRho0
        eQ = self._a0+eInputs[0]*self._a1
        iQ = self._a0

        idM = np.identity(self._a0.shape[0])
        for n in xrange(1, eInputs.size):
            if n%1000==0:
                print("Integration step %d (out of %d)" % (n, eInputs.size))
                sys.stdout.flush()
            eRhos[:, n] = (idM+self._dt*eQ).dot(eRhos[:, n-1])
            iRhos[:, n] = (idM+self._dt*iQ).dot(iRhos[:, n-1])
            eRs[n] = self._dv*eInputs[n]*self._reversedQs.dot(eRhos[:, n])
            iRs[n] = self._dv*wEI*eRs[n-1]*self._reversedQs.dot(iRhos[:, n])
            eQ = self._a0+eInputs[n]*self._a1-wIE*iRs[n-1]*self._a2
            iQ = self._a0+wEI*eRs[n-1]*self._a1
        return(eRhos, iRhos, eRs, iRs)

