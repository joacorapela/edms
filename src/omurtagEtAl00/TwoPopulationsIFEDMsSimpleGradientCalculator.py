
import sys
import pdb
import numpy as np
import numpy.linalg as la

class TwoPopulationsIFEDMsSimpleGradientCalculator:

    def __init__(self, a0, a1, a2, dt, dv, reversedQs):
        self._a0 = a0
        self._a1 = a1
        self._a2 = a2
        self._dt = dt
        self._dv = dv
        self._reversedQs = reversedQs

    def deriv(self, w12, w21, y1s, y2s, r1s, r2s, rho1s, rho2s, eInputs, 
                    ysSigma2):
        dRho1W12 = np.zeros(self._a0.shape[0])
        dRho1W21 = np.zeros(self._a0.shape[0])
        dRho2W12 = np.zeros(self._a0.shape[0])
        dRho2W21 = np.zeros(self._a0.shape[0])
        dR1W12 = 0
        dR1W21 = 0
        dR2W12 = 0
        dR2W21 = 0
        Q1 = self._a0 + eInputs[0]*self._a1
        Q2 = self._a0
        dQ1W12 = np.zeros(self._a1.shape)
        dQ1W21 = np.zeros(self._a1.shape)
        dQ2W12 = np.zeros(self._a1.shape)
        dQ2W21 = np.zeros(self._a1.shape)
        idM = np.identity(self._a0.shape[0])
        dLLW12 = dLLW21 = 0

        for n in xrange(1, eInputs.size):
            if n%1000==0:
                print("Gradient calculation step %d (out of %d)" % (n, eInputs.size))
                sys.stdout.flush()
            # The order of the following blocks is crucial
            # 1) dRho[n] depends on Q[n-1] and dQ[n-1], so dRho should come before dQ and Q
            # 2) dQ[n] depends on dR[n-1], so dQ should come before dR
            # 3) dR2[n] depends on dR1[n-1], so dR2 should come before dR1
            # 4) dR[n] depends on dRho[n], so dR should come after dRho

            dRho1W12 = self._dt*dQ1W12.dot(rho1s[:, n-1])+(idM+self._dt*Q1).dot(dRho1W12)
            dRho1W21 = self._dt*dQ1W21.dot(rho1s[:, n-1])+(idM+self._dt*Q1).dot(dRho1W21)
            dRho2W12 = self._dt*dQ2W12.dot(rho2s[:, n-1])+(idM+self._dt*Q2).dot(dRho2W12)
            dRho2W21 = self._dt*dQ2W21.dot(rho2s[:, n-1])+(idM+self._dt*Q2).dot(dRho2W21)

            Q1 = self._a0+eInputs[n]*self._a1-w21*r2s[n-1]*self._a2
            Q2 = self._a0+w12*r1s[n-1]*self._a1

            dQ1W12 = -w21*dR2W12*self._a2
            dQ1W21 = -(r2s[n-1]+w21*dR2W21)*self._a2
            dQ2W12 = (r1s[n-1]+w12*dR1W12)*self._a1
            dQ2W21 = w12*dR1W21*self._a1

            dR2W12 = self._dv*((r1s[n-1]+w12*dR1W12)*self._reversedQs.dot(rho2s[:, n])+
                         w12*r1s[n-1]*self._reversedQs.dot(dRho2W12))
            dR2W21 = self._dv*w12*(dR1W21*self._reversedQs.dot(rho2s[:, n])+
                              r1s[n-1]*self._reversedQs.dot(dRho2W21))
            dR1W12 = self._dv*eInputs[n]*self._reversedQs.dot(dRho1W12)
            dR1W21 = self._dv*eInputs[n]*self._reversedQs.dot(dRho1W21)

            dLLW12 = dLLW12 + (y1s[n]-r1s[n])*dR1W12 + \
                                    (y2s[n]-r2s[n])*dR2W12
            dLLW21 = dLLW21 + (y1s[n]-r1s[n])*dR1W21 + \
                                    (y2s[n]-r2s[n])*dR2W21

        dLLW12 = dLLW12/ysSigma2
        dLLW21 = dLLW21/ysSigma2
        return(dLLW12, dLLW21)

