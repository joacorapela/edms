
import numpy as np

class TwoPopulationsIFEnsembleDensitySimpleManualIntegrator:

    def integrate(self, eRho0, iRho0, wEI, wIE, eInputs, dt, dv, a0, a1, a2,
                        reversedQs):
        eR0 = dv*eInputs[0]*reversedQs.dot(eRho0)
        iR0 = 0
        eQ0 = a0+eInputs[0]*a1
        iQ0 = a0

        idM = np.identity(a0.shape[0])

        eRho1 = (idM + dt*eQ0).dot(eRho0)
        iRho1 = (idM + dt*iQ0).dot(iRho0)
        eR1 = dv*eInputs[1]*reversedQs.dot(eRho1)
        iR1 = dv*wEI*eR0*reversedQs.dot(iRho1)
        eQ1 = a0+eInputs[1]*a1
        iQ1 = a0+wEI*eR0*a1

        eRho2 = (idM+dt*eQ1).dot(eRho1)
        iRho2 = (idM+dt*iQ1).dot(iRho1)
        eR2 = dv*eInputs[2]*reversedQs.dot(eRho2)
        iR2 = dv*wEI*eR1*reversedQs.dot(iRho2)
        eQ2 = a0+eInputs[2]*a1-wIE*iR1*a2
        iQ2 = a0+wEI*eR1*a1

        eRhos = np.column_stack((eRho0, eRho1, eRho2))
        iRhos = np.column_stack((iRho0, iRho1, iRho2))
        eRs = np.array((eR0, eR1, eR2))
        iRs = np.array((iR0, iR1, iR2))
        return(eRhos, iRhos, eRs, iRs)

