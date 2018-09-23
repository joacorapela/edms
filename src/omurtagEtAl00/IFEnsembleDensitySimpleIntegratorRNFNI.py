
import sys
import numpy as np
import pdb

class IFEnsembleDensitySimpleIntegratorRNFNI:

    def __init__(self, a0, a1, dt, dv, reversedQs):
        self._a0 = a0
        self._a1 = a1
        self._dt =dt
        self._dv =dv
        self._reversedQs =reversedQs

    def integrate(self, rho0, inputs):
        rs = np.empty(inputs.size)
        rs[0] = self._dv*inputs[0]*self._reversedQs.dot(rho0)
        rhos = np.empty((self._a0.shape[0], inputs.size))
        rhos[:, 0] = rho0
        Q = self._a0+inputs[0]*self._a1

        idM = np.identity(self._a0.shape[0])
        for n in xrange(1, inputs.size):
            if n%1000==0:
                print("Integration step %d (out of %d)" % (n, inputs.size))
                sys.stdout.flush()
            rhos[:, n] = (idM+self._dt*Q).dot(rhos[:, n-1])
            rs[n] = self._dv*inputs[n]*self._reversedQs.dot(rhos[:, n])
            Q = self._a0+inputs[n]*self._a1
        return(rhos, rs)

