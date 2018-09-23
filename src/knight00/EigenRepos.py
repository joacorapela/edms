
import numpy as np
import pdb
import matplotlib.pyplot as plt
from myMath import multiplyMatrixColsByScalars
from myMath import normalizedCrossCorrelation

class EigenRepos:

    def computeEigenDecomposition(self, diffMatrix, prevEVecs=None):
        eVals, eVecs, aEVecs, sortIndices, flippedEVecs = \
         self._computeUnnormalizedEigenDecomposition(diffMatrix=diffMatrix,
                                                  prevEVecs=prevEVecs)
        normalizationIndices = self._normalizeEigenvectors(eVecs=eVecs, 
                                                            aEVecs=aEVecs)

        # Lets enforce eVals[0]=0
        eVals[0] = 0
        return(eVals, eVecs, aEVecs, sortIndices, flippedEVecs,
                      normalizationIndices)

#     def _computeDEvalsAndDEScaleMatrix(self, dMatrix, eVals, eVecs, aEVecs,
#                                              normalizationIndices):
#         # aEVDMatrixEV(k, l)=y_k^* A' x_l
#         aEVDMatrixEV = aEVecs.transpose().conjugate().dot(dMatrix.dot(eVecs))
#         dEVals = np.diag(aEVDMatrixEV)
#         # eValsDifMatrix(k, l)=lambda_k-lambda_l if k!=l: 
#         # eValsDifMatrix(k, k)=1.0
#         eValsDifMatrix = np.outer(eVals, np.ones(eVals.size))-\
#                            np.outer(np.ones(eVals.size), eVals)
#         np.fill_diagonal(eValsDifMatrix, 1.0)
#         # dEScaleMatrix is matrix C in Eqs. 4, 15, and 19 in
#         # vanDerAa06-derivativeEigen.pdf
#         # dEVecs=eVecs %*% dEScaleMatrix
#         dEScaleMatrix = -aEVDMatrixEV/eValsDifMatrix
#         # Setting the diagonal of dEScaleMatrix to zero is a trick to
#         # compyute the summation in Eq. 19 as an inner product
#         np.fill_diagonal(dEScaleMatrix, 0.0)
#         tmpDiagDEVecsScaleMatrix = np.empty(dEScaleMatrix.shape[1],
#                                              dtype=complex)
#         for k in xrange(tmpDiagDEVecsScaleMatrix.size):
#             tmpDiagDEVecsScaleMatrix[k] = \
#              -np.dot(eVecs[normalizationIndices[k], :], dEScaleMatrix[:, k])
#         np.fill_diagonal(dEScaleMatrix, tmpDiagDEVecsScaleMatrix)
#         return(dEVals, dEScaleMatrix)

#     def computeEigenDerivs(self, dMatrix, eVals, eVecs, aEVecs,
#                                  normalizationIndices):
#         # Following the elegant presentation by vanDerAa06-derivativeEigen.pdf
#         evals, dEScaleMatrix = \
#          self._computeDEvalsAndDEScaleMatrix(dMatrix=dMatrix, 
#                                               eVals=eVals,
#                                               eVecs=eVecs,
#                                               aEVecs=aEVecs,
#                                               normalizationIndices=
#                                                normalizationIndices)
#                                              
#         dEVecs = eVecs.dot(dEScaleMatrix)
#         # The derivative of the adjoint eigenvectors is not given in
#         # vanDerAa06-derivativeEigen.pdf, but it is not difficult to derive.
#         # See page 3 on my notes on Eigenvectors and Eigenvalues derivaties.
#         dAEVecs = -aEVecs.dot(dEScaleMatrix.transpose().conjugate())
#         return(dEVals, dEVecs, dAEVecs)

    def _computeUnnormalizedEigenDecomposition(self, diffMatrix, 
                                                     prevEVecs=None):
        eigRes = np.linalg.eig(diffMatrix)
        sortIndices, flippedEVecs = \
         self._sortAndFlipEigRes(eigRes=eigRes, prevEVecs=prevEVecs)
        aEVecs = np.linalg.pinv(eigRes[1]).transpose().conjugate()
        return(eigRes[0], eigRes[1], aEVecs, sortIndices, flippedEVecs)

    def _normalizeEigenvectors(self, eVecs, aEVecs):
        normalizationIndices = self._getNormalizationIndices(eVecs=eVecs,
                                                              aEVecs=aEVecs)
        normalizationConstants = 1.0/abs(eVecs[normalizationIndices, 
                                                np.arange(eVecs.shape[1])])
        # begin to delete
        if np.any(normalizationConstants<0):
            raise ValueError("some normalizationConstants<0")
        # end to delete
        eVecs[:,:] = \
         multiplyMatrixColsByScalars(m=eVecs, s=normalizationConstants)
        aEVecs[:,:] = \
         multiplyMatrixColsByScalars(m=aEVecs, s=1.0/normalizationConstants)
        return(normalizationIndices)

    def _sortAndFlipEigRes(self, eigRes, prevEVecs=None):
        if(prevEVecs is None):
            preliminarySortIndices = np.argsort(-eigRes[0].real)
            eigRes[0][:] = eigRes[0][preliminarySortIndices]
            eigRes[1][:,:] = eigRes[1][:,preliminarySortIndices]
            sortIndices = np.arange(eigRes[1].shape[1])
            flippedEVecs =  np.empty(eigRes[1].shape[1])
            flippedEVecs[:] = False
        else:
            preliminarySortIndices = np.argsort(-eigRes[0].real)
            eigRes[0][:] = eigRes[0][preliminarySortIndices]
            eigRes[1][:,:] = eigRes[1][:,preliminarySortIndices]

            curEVecsMatchingPrevEVecs, maxCrossCorForPrevEVecs = \
             self._findIndicesEVecsMatchingPrevEVecs(eVecs=eigRes[1], 
                                                      prevEVecs=prevEVecs)
            sortIndices = curEVecsMatchingPrevEVecs
            flippedEVecs = \
             self._flipEigenvectors(eVecs=eigRes[1], prevEVecs=prevEVecs,
                                                     curEVecsMatchingPrevEVecs=
                                                      curEVecsMatchingPrevEVecs)
        eigRes[0][:] = eigRes[0][sortIndices]
        eigRes[1][:,:] = eigRes[1][:,sortIndices]
        flippedEVecs = flippedEVecs[sortIndices]
        return(sortIndices, flippedEVecs)

    def _findIndicesEVecsMatchingPrevEVecs(self, eVecs, prevEVecs):
        nEigenvalues = eVecs.shape[1]
        curEVecsMatchingPrevEVecs = np.empty(nEigenvalues, dtype=int)
        maxCrossCorForPrevEVecs  = np.empty(nEigenvalues, dtype=complex)
        curEVecsIndicesSearchList = range(nEigenvalues)
        for prevEVecIndex in xrange(prevEVecs.shape[1]):
            prevEVec = prevEVecs[:, prevEVecIndex]
            crossCorsForPrevEVec = np.empty(len(curEVecsIndicesSearchList), 
                                             dtype=complex)
            for curEVecsSearchListIndex in \
                 xrange(len(curEVecsIndicesSearchList)):
                curEVecIndex = \
                 curEVecsIndicesSearchList[curEVecsSearchListIndex]
                curEVec = eVecs[:, curEVecIndex]
                crossCorsForPrevEVec[curEVecsSearchListIndex] = \
                 normalizedCrossCorrelation(prevEVec, curEVec)
            maxCrossCorForPrevEVecIndex = \
             np.argmax(abs(crossCorsForPrevEVec))
            maxCrossCorForPrevEVecs[prevEVecIndex] = \
             crossCorsForPrevEVec[maxCrossCorForPrevEVecIndex]
            curEVecsMatchingPrevEVecs[prevEVecIndex] = \
             curEVecsIndicesSearchList[maxCrossCorForPrevEVecIndex]
            curEVecsIndicesSearchList.pop(maxCrossCorForPrevEVecIndex)
        return(curEVecsMatchingPrevEVecs, maxCrossCorForPrevEVecs)

    def _flipEigenvectors(self, eVecs, prevEVecs, curEVecsMatchingPrevEVecs):
        # Flip an eVec if it is negatively correlated with the best matching 
        # prevEVec
        flippedEVecs =  np.empty(eVecs.shape[1])
        flippedEVecs[:] = False
        for prevEVecIndex in xrange(eVecs.shape[1]):
            curEVecIndex = curEVecsMatchingPrevEVecs[prevEVecIndex]
            prevEVec = prevEVecs[:, prevEVecIndex]
            curEVec = eVecs[:, curEVecIndex]
            difPrevEVec = self._differentiateVector(aVector=prevEVec)
            difCurEVec = self._differentiateVector(aVector=curEVec)
            normalizedCrossCor = normalizedCrossCorrelation(difPrevEVec, 
                                                             difCurEVec)
            if normalizedCrossCor<0:
#                 plt.figure(); plt.plot(prevEVec, label="Previous"); plt.plot(curEVec, label="Current"); plt.legend(); plt.show()
                eVecs[:, curEVecIndex] = -curEVec
                flippedEVecs[curEVecIndex] = True
        return(flippedEVecs)

    def _differentiateVector(self, aVector):
        return(aVector[1:aVector.size]-aVector[:(aVector.size-1)])

    def _getNormalizationIndices(self, eVecs, aEVecs):
#         normalizationIndices = np.argmax(eVecs, axis=0)
        normalizationIndices = np.argmax(abs(eVecs)*abs(aEVecs), axis=0)
#         normalizationIndices = np.argmax(abs(eVecs), axis=0)
        return(normalizationIndices)

