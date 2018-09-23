
class IFEDMWithInputLayer(object):
    def __init__(self, ifEDM, eInputLayer, iInputLayer):
        self._ifEDM = ifEDM
        self._eInputLayer = eInputLayer
        self._iInputLayer = iInputLayer

    def prepareToIntegrate(self, t0, tf, dt):
        self._ifEDM.prepareToIntegrate(t0=t0, 
                                        tf=tf, 
                                        dt=dt,
                                        eExternalInput=\
                                         self._eInputLayer.getSpikeRate,
                                        iExternalInput=\
                                         self._iInputLayer.getSpikeRate)

    def setInitialValue(self, rho0):
        self._ifEDM.setInitialValue(rho0=rho0)

    def integrate(self, dtSaveRhos=None):
        return(self._ifEDM.integrate(dtSaveRhos=dtSaveRhos))

    def getEExternalInputHist(self):
        return(self._ifEDM.getEExternalInputHist())

    def getIExternalInputHist(self):
        return(self._ifEDM.getIExternalInputHist())

    def getEFeedbackInputHist(self):
        return(self._ifEDM.getEFeedbackInputHist())

    def getIFeedbackInputHist(self):
        return(self._ifEDM.getIFeedbackInputHist())


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

    def getEL(self):
        return(self._eInputLayer.getRectification().getL())

    def getEK(self):
        return(self._eInputLayer.getRectification().getK())

    def getEX0(self):
        return(self._eInputLayer.getRectification().getX0())

    def getIL(self):
        return(self._iInputLayer.getRectification().getL())

    def getIK(self):
        return(self._iInputLayer.getRectification().getK())

    def getIX0(self):
        return(self._iInputLayer.getRectification().getX0())

    def getEFilter(self):
        return(self._eInputLayer.getFilter())

    def getIFilter(self):
        return(self._iInputLayer.getFilter())


    def setLeakage(self, leakage):
        self._ifEDM.setLeakage(leakage=leakage)

    def setG(self, g):
        self._ifEDM.setNInputsPerNeuron(nInputsPerNeuron=g)

    def setF(self, f):
        self._ifEDM.setFracExcitatoryNeurons(fracExcitatoryNeurons=f)

    def setEL(self, eL):
        self._eInputLayer.getRectification().setL(l=eL)

    def setEK(self, eK):
        self._eInputLayer.getRectification().setK(k=eK)

    def setEX0(self, eX0):
        self._eInputLayer.getRectification().setX0(x0=eX0)

    def setIL(self, iL):
        self._iInputLayer.getRectification().setL(l=iL)

    def setIK(self, eK):
        self._iInputLayer.getRectification().setK(k=iK)

    def setIX0(self, eX0):
        self._iInputLayer.getRectification().setX0(x0=iX0)

    def setEFilter(self, filter):
        self._eInputLayer.setFilter(filter=filter)

    def setIFilter(self, filter):
        self._iInputLayer.setFilter(filter=filter)

