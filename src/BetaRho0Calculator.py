
import numpy as np
import scipy.stats as stats
import scipy.special as special

class BetaRho0Calculator(object):
    def __init__(self, nVSteps):
        self._vs = (np.linspace(1, nVSteps, nVSteps)-0.5)/nVSteps

    def getVs(self):
        return(self._vs)

    def getRho0(self, a, b):
        rho0 = stats.beta.pdf(self._vs, a, b)
        return(rho0)

    def getDerivRho0(self, a, b):
        rho0 = self.getRho0(a=a, b=b)
        lnVs = np.log(self._vs)
        lnOneMinusVs = np.log(1-self._vs)

        B = special.beta(a, b)
        psiAplusB = special.psi(a+b)
        derivBA = B*(special.psi(a)-psiAplusB)
        derivBB = B*(special.psi(b)-psiAplusB)

        derivRho0A = rho0*(lnVs-derivBA/B)
        derivRho0B = rho0*(lnOneMinusVs-derivBB/B)

        return(derivRho0A, derivRho0B)
