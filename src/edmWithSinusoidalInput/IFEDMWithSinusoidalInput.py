
class IFEDMWithSinusoidalInput(object):
    def __init__(self, ifEDM, eSinusoidal, iSinusoidal):
        self._ifEDM = ifEDM
        self._eSinusoidal = eSinusoidal
        self._iSinusoidal = iSinusoidal

    def prepareToIntegrate(self, t0, tf, dt):
        self._ifEDM.prepareToIntegrate(t0=t0, 
                                        tf=tf, 
                                        dt=dt,
                                        eExternalInput=\
                                         self._eSinusoidal.eval,
                                        iExternalInput=\
                                         self._iSinusoidal.eval)


    def setInitialValue(self, rho0):
        self._ifEDM.setInitialValue(rho0=rho0)

    def integrate(self, dtSaveRhos=None, nStepsBtwPrintouts=1000):
        return(self._ifEDM.integrate(dtSaveRhos=dtSaveRhos,
                                      nStepsBtwPrintouts=nStepsBtwPrintouts))

    def getEExternalInputHist(self):
        return(self._ifEDM.getEExternalInputHist())

    def getIExternalInputHist(self):
        return(self._ifEDM.getIExternalInputHist())

    def getEFeedbackInputHist(self):
        return(self._ifEDM.getEFeedbackInputHist())

    def getIFeedbackInputHist(self):
        return(self._ifEDM.getIFeedbackInputHist())


    def getIFEDM(self):
        return(self._ifEDM)

    def getA0Tilde(self):
        return(self._ifEDM.getA0Tilde())

    def getA1(self):
        return(self._ifEDM.getA1())

    def getA2(self):
        return(self._ifEDM.getA2())

    def getReversedQs(self):
        return(self._ifEDM.getReversedQs())

    def getLeakage(self):
        return(self._ifEDM.getLeakage())

    def getG(self):
        return(self._ifEDM.getNInputsPerNeuron())

    def getF(self):
        return(self._ifEDM.getFracExcitatoryNeurons())

    def getEDC(self):
        return(self._eSinusoidal.getDC())

    def getEAmpl(self):
        return(self._eSinusoidal.getAmpl())

    def getEFreq(self):
        return(self._eSinusoidal.getFreq())

    def getEPhase(self):
        return(self._eSinusoidal.getPhase())

    def getIDC(self):
        return(self._iSinusoidal.getDC())

    def getIAmpl(self):
        return(self._iSinusoidal.getAmpl())

    def getIFreq(self):
        return(self._iSinusoidal.getFreq())

    def getIPhase(self):
        return(self._iSinusoidal.getPhase())

    def setLeakage(self, leakage):
        self._ifEDM.setLeakage(leakage=leakage)

    def setG(self, g):
        self._ifEDM.setNInputsPerNeuron(nInputsPerNeuron=g)

    def setF(self, f):
        self._ifEDM.setFracExcitatoryNeurons(fracExcitatoryNeurons=f)

    def setEDC(self, eDC):
        self._eSinusoidal.setDC(dc=eDC)

    def setEAmpl(self, eAmpl):
        self._eSinusoidal.setAmpl(ampl=eAmpl)

    def setEFreq(self, eFreq):
        self._eSinusoidal.setFreq(freq=eFreq)

    def setEPhase(self, ePhase):
        self._eSinusoidal.setPhase(phase=ePhase)

    def setIDC(self, iDC):
        self._iSinusoidal.setDC(dc=iDC)

    def setIAmpl(self, iAmpl):
        self._iSinusoidal.setAmpl(ampl=iAmpl)

    def setIFreq(self, iFreq):
        self._iSinusoidal.setFreq(freq=iFreq)

    def setIPhase(self, iPhase):
        self._iSinusoidal.setPhase(phase=iPhase)

