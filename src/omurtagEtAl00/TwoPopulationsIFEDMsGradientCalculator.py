
import pdb
import numpy as np
import numpy.linalg as la

class TwoPopulationsIFEDMsGradientCalculator:

    def __init__(self, a0, a1, a2, sigma0, dt, dv, reversedQs):
        self._a0 = a0
        self._a1 = a1
        self._a2 = a2
        self._sigma0 = sigma0
        self._dt = dt
        self._dv = dv
        self._reversedQs = reversedQs

    def deriv(self, w12, w21, times1, times2, y1s, y2s, r1s, r2s, rho1s, rho2s):
        # y?s and r?s should be sampled at dtSaveRhos
        dRho1W12 = dRho1W21 = dRho2W12 = dRho2W21=np.zeros(self._a0.shape[0])
        dr1W12 = dr1W21 = dr2W12 = dr2W21 = 0
        idM = np.identity(self._a0.shape[0])
        dLLW12 = dLLW21 = 0

        dr1W12s = np.empty(y1s.size)
        dr1W21s = np.empty(y1s.size)
        dr2W12s = np.empty(y1s.size)
        dr2W21s = np.empty(y1s.size)
        dr1W12s[0] = dr1W12
        dr1W21s[0] = dr1W21
        dr2W12s[0] = dr2W12
        dr2W21s[0] = dr2W21
        for n in xrange(1, y1s.size):
            if n%100==0:
                print("Processing time %.05f out of %.02f" % \
                      (times1[n], times1[y1s.size-1]))
            dR1W12 = self._dt*w21*dr2W12*self._a2
            dR1W21 = self._dt*(r2s[n-1]+w21*dr2W21)*self._a2
            dR2W12 = -self._dt*(r1s[n-1]+w12*dr1W12)*self._a1
            dR2W21 = -self._dt*w12*dr1W21*self._a1

            sigma0AtTn = self._sigma0(t=times1[n])
            invR1 = la.inv(idM-self._dt*(self._a0+sigma0AtTn*self._a1-
                                                  w21*r2s[n-1]*self._a2))
            invR2 = la.inv(idM-self._dt*(self._a0+w12*r1s[n-1]*self._a1))

            dInvR1W12 = invR1.dot(dR1W12).dot(invR1)
            dInvR1W21 = invR1.dot(dR1W21).dot(invR1)
            dInvR2W12 = invR2.dot(dR2W12).dot(invR2)
            dInvR2W21 = invR2.dot(dR2W21).dot(invR2)

            dRho1W12 = dInvR1W12.dot(rho1s[:, n-1]) + invR1.dot(dRho1W12)
            dRho1W21 = dInvR1W21.dot(rho1s[:, n-1]) + invR1.dot(dRho1W21)
            dRho2W12 = dInvR2W12.dot(rho2s[:, n-1]) + invR2.dot(dRho2W12)
            dRho2W21 = dInvR2W21.dot(rho2s[:, n-1]) + invR2.dot(dRho2W21)

            dr2W12 = self._dv*((r1s[n-1]+w12*dr1W12)*
                         self._reversedQs.dot(rho2s[:, n])+
                         w12*r1s[n-1]*self._reversedQs.dot(dRho2W12))
            dr2W21 = self._dv*w12*(dr1W21*self._reversedQs.dot(rho2s[:, n])+
                              r1s[n-1]*self._reversedQs.dot(dRho2W21))
            dr1W12 = self._dv*sigma0AtTn*self._reversedQs.dot(dRho1W12)
            dr1W21 = self._dv*sigma0AtTn*self._reversedQs.dot(dRho1W21)

            dr1W12s[n] = dr1W12
            dr1W21s[n] = dr1W21
            dr2W12s[n] = dr2W12
            dr2W21s[n] = dr2W21

            dLLW12 = dLLW12 + (y1s[n]-r1s[n])*dr1W12 + \
                                    (y2s[n]-r2s[n])*dr2W12
            dLLW21 = dLLW21 + (y1s[n]-r1s[n])*dr1W21 + \
                                    (y2s[n]-r2s[n])*dr2W21
        return(dLLW12, dLLW21)

