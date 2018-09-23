
import math
import numpy as np

class Sinusoidal:

    def __init__(self, dc, ampl, freq, phase):
        self._dc = dc
        self._ampl = ampl
        self._freq = freq
        self._phase = phase

    def eval(self, t):
        answer = self._dc+\
                 (1+self._ampl*np.sin(2*math.pi*self._freq*t+self._phase))
        return(answer)

    def getDC(self):
        return(self._dc)

    def getAmpl(self):
        return(self._ampl)

    def getFreq(self):
        return(self._freq)

    def getPhase(self):
        return(self._phase)

    def setDC(self, dc):
        self._dc = dc

    def setAmpl(self, ampl):
        self._ampl = ampl

    def setFreq(self, freq):
        self._freq = freq

    def setPhase(self, phase):
        self._phase = phase

