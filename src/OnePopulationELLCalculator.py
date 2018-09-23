
import numpy as np
import pdb
from edmsMath import calculateELL

from edmsMath import calculateLL
class OnePopulationELLCalculator:

    def calculateELLsForParams(self, ys, rsForParams, ysSigma):
        lls = []
        for paramIndex in xrange(rsForParams.shape[0]):
            rsForParam = \
             np.reshape(rsForParams[paramIndex, :],
                         rsForParams.shape[1:])
            lls.append(calculateELL(ys=ys, rs=rsForParam, ysSigma=ysSigma))
        return(lls)

