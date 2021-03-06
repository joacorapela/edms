
import pdb
import numpy as np
from IFEnsembleDensityIntegratorRWFWI import IFEnsembleDensityIntegratorRWFWI
from IFEDMWithInputLayer import IFEDMWithInputLayer
from StimuliSourceFromFile import StimuliSourceFromFile
from utilsMath import Logistic
# from utilsMath import LinearFunction
from InputLayer import InputLayer

class SpikeRatesForParamsCalculatorOfOneEDMWithInputLayer:
    def __init__(self, hMu, hSigma, kappaMu, kappaSigma,
                       t0, tf, dt, nVSteps, rho0, 
                       stimuliFilename):
        self._hMu = hMu
        self._hSigma = hSigma
        self._kappaMu = kappaMu
        self._kappaSigma = kappaSigma
        self._t0 = t0
        self._tf = tf
        self._dt = dt
        self._nVSteps = nVSteps
        self._rho0 = rho0
        self._stimuliFilename = stimuliFilename

    def calculateSpikeRatesForParams(self, params):
        nTSteps = round((self._tf-self._t0)/self._dt)
        spikeRatesForParams = np.empty((len(params), nTSteps))
        for i in xrange(len(params)):
            times, spikeRatesForOnseSetParams = \
             self._calculateSpikeRatesForOneSetParams(oneSetParams=params[i])
            spikeRatesForParams[i, :] = spikeRatesForOnseSetParams
        return(times, spikeRatesForParams)

    def _calculateSpikeRatesForOneSetParams(self, oneSetParams):
        leakage = oneSetParams[0]
        g = oneSetParams[1]
        f = oneSetParams[2]
        lE = oneSetParams[3]
        kE = oneSetParams[4]
        x0E = oneSetParams[5]
        lI = oneSetParams[6]
        kI = oneSetParams[7]
        x0I = oneSetParams[8]
        filterLength = (len(oneSetParams)-9)/2
        eFilter = np.array(oneSetParams[9:(9+filterLength)])
        iFilter = np.array(oneSetParams[(9+filterLength):(9+2*filterLength)])
        print("Calculating spike rates for leakage=%.02f, g=%d, f=%.02f"%(leakage, g, f))
        print("                            lE=%.02f, kE=%f, x0E=%.02f"%(lE, kE, x0E))
        print("                            lI=%.02f, kI=%f, x0I=%.02f"%(lI, kI, x0I))
        print("Excitatory filter: %s"%(str(eFilter)))
        print("Inhibitory filter: %s"%(str(iFilter)))

        ifEDM = IFEnsembleDensityIntegratorRWFWI(nVSteps=self._nVSteps, 
                                                  leakage=leakage, 
                                                  hMu=self._hMu, 
                                                  hSigma=self._hSigma, 
                                                  kappaMu=self._kappaMu,
                                                  kappaSigma=self._kappaSigma,
                                                  fracExcitatoryNeurons=f,
                                                  nInputsPerNeuron=g)
        stimSource = StimuliSourceFromFile(stimuliFilename=
                                             self._stimuliFilename, 
                                            t0=self._t0, tf=self._tf, 
                                            dt=self._dt)
        eRectification = Logistic(k=kE, x0=x0E, l=lE)
        iRectification = Logistic(k=kI, x0=x0I, l=lI)

        eInputLayer = InputLayer(stimuliSource=stimSource, 
                                  rectification=eRectification,
                                  filter=eFilter,
                                  baselineInput=0.0)
        iInputLayer = InputLayer(stimuliSource=stimSource,
                                  rectification=iRectification,
                                  filter=iFilter,
                                  baselineInput=0.0)

        ifEDMWithInputLayer = IFEDMWithInputLayer(ifEDM=ifEDM,
                                                   eInputLayer=eInputLayer,
                                                   iInputLayer=iInputLayer)
        ifEDMWithInputLayer.prepareToIntegrate(t0=self._t0, tf=self._tf,
                                                            dt=self._dt)

        ifEDMWithInputLayer.setInitialValue(rho0=self._rho0)
        ts, rhos, spikeRates, saveRhosTimeDSFactor = \
         ifEDMWithInputLayer.integrate()
        return(ts, spikeRates)

