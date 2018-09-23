
import numpy as np
import pdb
import matplotlib.pyplot as plt
from myMath import multiplyMatrixColsByScalars
from myMath import normalizedCrossCorrelation

class LargeEigenReposForOneStim:

    def __init__(self, a0, a1, minStim, maxStim, stepStim, qs):
        reversedQs = qs[-1::-1]
        self._stimuli = np.arange(start=minStim, stop=maxStim, step=stepStim)
        self._eVals = np.empty((a0.shape[0], self._stimuli.size), dtype=complex)
        self._eVecs = []
        self._aEVecs = []
        self._dAEVecs = []

        derDiffMatrix = a1
        prevEVecs = None

        for i in xrange(len(self._stimuli)):
            s = self._stimuli[i]
            print("Processing stimulus %f" % s)
            diffMatrix = a0+s*a1
            eVals, eVecs, aEVecs = self.getEigenResults(diffMatrix=diffMatrix,
                                                         prevEVecs=prevEVecs)
            self._eVals[:, i] = eVals
            self._eVecs.append(eVecs)
            self._aEVecs.append(aEVecs)

            dEVals, dEVecs, dAEVecs = self._getEigenDerivs(dMatrix= derDiffMatrix, eVals=eVals, eVecs=eVecs, aEVecs=aEVecs)
            self._dAEVecs.append(dAEVecs)

            prevEVecs = eVecs

    def getEVals(self, s):
        stimIndex = np.argmin(abs(self._stimuli-s))
        return(self._eVals[:, stimIndex])

    def getEVecs(self, s):
        stimIndex = np.argmin(abs(self._stimuli-s))
        return(self._eVecs[stimIndex])

    def getAEVecs(self, s):
        stimIndex = np.argmin(abs(self._stimuli-s))
        return(self._aEVecs[stimIndex])

    def getDAEVecs(self, s):
        stimIndex = np.argmin(abs(self._stimuli-s))
        return(self._dAEVecs[stimIndex])

    def _getEigenDerivs(self, dMatrix, eVals, eVecs, aEVecs):
        # Following the elegant presentation by vanDerAa06-derivativeEigen.pdf

        # aEVDMatrixEV(k, l)=y_k^* A' x_l
        aEVDMatrixEV = aEVecs.transpose().conjugate().dot(dMatrix.dot(eVecs))
        dEVals = np.diag(aEVDMatrixEV)
        # eValsDifMatrix(k, l)=lambda_k-lambda_l if k!=l: 
        # eValsDifMatrix(k, k)=1.0
        eValsDifMatrix = np.outer(eVals, np.ones(eVals.size))-\
                           np.outer(np.ones(eVals.size), eVals)
        np.fill_diagonal(eValsDifMatrix, 1.0)
        # dEVecsScaleMatrix is matrix C in Eqs. 4, 15, and 19 in
        # vanDerAa06-derivativeEigen.pdf
        # dEVecs=eVecs %*% dEVecsScaleMatrix
        dEVecsScaleMatrix = -aEVDMatrixEV/eValsDifMatrix
        # Setting the diagonal of dEVecsScaleMatrix to zero is a trick to
        # compyute the summation in Eq. 19 as an inner product
        np.fill_diagonal(dEVecsScaleMatrix, 0.0)
        tmpDiagDEVecsScaleMatrix = np.empty(dEVecsScaleMatrix.shape[1],
                                             dtype=complex)
        for k in xrange(tmpDiagDEVecsScaleMatrix.size):
            tmpDiagDEVecsScaleMatrix[k] = \
             -np.dot(eVecs[self._normalizationIndices[k], :], 
                      dEVecsScaleMatrix[:, k])
        np.fill_diagonal(dEVecsScaleMatrix, tmpDiagDEVecsScaleMatrix)
        dEVecs = eVecs.dot(dEVecsScaleMatrix)
        # The derivative of the adjoint eigenvectors is not given in
        # vanDerAa06-derivativeEigen.pdf, but it is not difficult to derive.
        # See page 3 on my notes on Eigenvectors and Eigenvalues derivaties.
        dAEVecs = -aEVecs.dot(dEVecsScaleMatrix.transpose().conjugate())
        return(dEVals, dEVecs, dAEVecs)

    def getEigenResults(self, diffMatrix, prevEVecs=None):
        eVals, eVecs = self._getUnscaledEigenResults(diffMatrix=diffMatrix,
                                                      prevEVecs=prevEVecs)
#         if prevEVecs is not None:
#             i=6; plt.plot(eVecs[:, i], label="current"); plt.plot(prevEVecs[:, i], label="previous"); plt.title("Eigenvector %d" % i); plt.legend(); plt.show()
#         pdb.set_trace()
        self._scaleEigenvectors(eVecs=eVecs)
#         if prevEVecs is not None:
#             i=6; plt.plot(eVecs[:, i], label="current"); plt.plot(prevEVecs[:, i], label="previous"); plt.title("Eigenvector %d" % i); plt.legend(); plt.show()
#         pdb.set_trace()
        aEVecs = np.linalg.inv(eVecs).transpose().conjugate()
        return(eVals, eVecs, aEVecs)

    def _getUnscaledEigenResults(self, diffMatrix, prevEVecs):
        eigRes = np.linalg.eig(diffMatrix)
        self._sortAndFlipEigRes(eigRes=eigRes, prevEVecs=prevEVecs)
        return(eigRes[0], eigRes[1])

    def _scaleEigenvectors(self, eVecs):
        self._normalizationIndices = self._getNormalizationIndices(eVecs=eVecs)
        normalizationConstants = 1.0/abs(eVecs[self._normalizationIndices, 
                                                np.arange(eVecs.shape[1])])
        eVecs[:,:] = \
         multiplyMatrixColsByScalars(m=eVecs, s=normalizationConstants)

    def _sortAndFlipEigRes(self, eigRes, prevEVecs):
        if(prevEVecs is None):
            sortIndices = np.argsort(-eigRes[0].real)
        else:
            nEigen = eigRes[0].size
            sortIndices = np.empty(nEigen, dtype=int)
            searchList = range(nEigen)
            for i in xrange(nEigen):
                prevEVec = prevEVecs[:, i]
                crossCors = np.empty(len(searchList), dtype=complex)
                for j in xrange(len(searchList)):
                    crossCors[j] = \
                     normalizedCrossCorrelation(eigRes[1][:, searchList[j]], 
                                                 prevEVec)
                maxIndex = np.argmax(abs(crossCors))
                sortIndices[i] = searchList[maxIndex]
                searchList.pop(maxIndex)
                # Flip the new eigenvector if it is negatively correlated with 
                # the best matching eigenvector in the previou time point
                if crossCors[maxIndex]<0:
                    eigRes[1][:, sortIndices[i]] = -eigRes[1][:, sortIndices[i]]
#                 plt.plot(eigRes[1][:, sortIndices[i]], label="current"); plt.plot(prevEVec, label="previous"); plt.title("Eigenvector %d" % i); plt.legend(); plt.show()
#                 pdb.set_trace()
        eigRes[0][:] = eigRes[0][sortIndices]
        eigRes[1][:,:] = eigRes[1][:,sortIndices]
#         if prevEVecs is not None:
#             i=6; plt.plot(eigRes[1][:,i], label="current"); plt.plot(prevEVecs[:,i], label="previous"); plt.title("EVec %d" % i); plt.legend(); plt.show()
#             pdb.set_trace()

    def _getNormalizationIndices(self, eVecs):
#         normalizationIndices = np.argmax(abs(eVecs)*abs(aEVecs), axis=0)
        normalizationIndices = np.argmax(abs(eVecs), axis=0)
        return(normalizationIndices)

