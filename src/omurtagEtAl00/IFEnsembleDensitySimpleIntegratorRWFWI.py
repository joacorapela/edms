
import sys
import numpy as np
import pdb

class IFEnsembleDensitySimpleIntegratorRWFWI:

    def __init__(self, a0, a1, a2, dt, dv, 
                       nInputsPerNeuron, fracExcitatoryNeurons,
                       reversedQs):
        self._a0 = a0
        self._a1 = a1
        self._a2 = a2
        self._dt =dt
        self._dv =dv
        self._nInputsPerNeuron = nInputsPerNeuron
        self._fracExcitatoryNeurons = fracExcitatoryNeurons
        self._reversedQs =reversedQs

    def integrate(self, rho0, eInputs, iInputs):
        rs = np.empty(eInputs.size)
        rs[0] = self._dv*eInputs[0]*self._reversedQs.dot(rho0)
        rhos = np.empty((self._a0.shape[0], eInputs.size))
        rhos[:, 0] = rho0
        Q = self._a0+\
             (eInputs[0]+self._nInputsPerNeuron*
                         self._fracExcitatoryNeurons*rs[0])*self._a1-\
             (iInputs[0]+self._nInputsPerNeuron*
                         (1-self._fracExcitatoryNeurons)*rs[0])*self._a2

        idM = np.identity(self._a0.shape[0])
        for n in xrange(1, eInputs.size):
            if n%1000==0:
                print("Integration step %d (out of %d)" % (n, eInputs.size))
                sys.stdout.flush()
            rhos[:, n] = (idM+self._dt*Q).dot(rhos[:, n-1])
            rs[n] = eInputs[n]*self._dv*self._reversedQs.dot(rhos[:, n])/\
                     1-self._nInputsPerNeuron*self._fracExcitatoryNeurons*\
                       self._dv*self._reversedQs.dot(rhos[:, n])
            Q = self._a0+\
                (eInputs[n]+self._nInputsPerNeuron*
                            self._fracExcitatoryNeurons*rs[n])*self._a1-\
                (iInputs[n]+self._nInputsPerNeuron*
                            (1-self._fracExcitatoryNeurons)*rs[n])*self._a2
        return(rhos, rs)

