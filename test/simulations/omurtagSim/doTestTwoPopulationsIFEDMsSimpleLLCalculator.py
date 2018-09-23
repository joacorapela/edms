
import sys
import numpy as np
import pdb
from LLCalculator import LLCalculator

def main(argv):
    sigma = 0.1
    dt = 1e-5
    t0 = 0.0
    tfSpikeRatesForParams = 0.25
    tfYs = 0.30

    spikesRatesForParamsFilename = \
     "results/spikeRatesForParamsOfTwoPopulationsT0%.02fTf%.02fDt%fWIEs5-50-5WEIs10-100-10.npz" % (t0, tfSpikeRatesForParams, dt)
#      "results/spikeRatesForParamsOfTwoPopulationsT0%.02fTf%0.02fDt%fEDM1IWs0-50-5EDM2EWs0-100-10.npz" % (t0, tfSpikeRatesForParams, dt)
    ysFilename = \
     "results/simpleYsForTwoPopulationsT0%.02fTf%.02fDt%fWEENoneWIE15WEI50WIINoneSigma%.02f.npz" % (t0, tfYs, dt, sigma)
#      "results/ysForTwoPopulationsT0%.02fTf%.02fDt%fEDM1EWNoneEDM1IW15EDM2EW50EDM2IWNoneSigma%.02f.npz" % (t0, tfYs, dt, sigma)
    resultsFilenamePattern = \
     "results/simpleLLsForParamsOfTwoPopulationsT0%.02fTf%.02fDt%fWIEs5-50-5WEIs10-100-10-sigma%.02f.npz" % (t0, tfSpikeRatesForParams, dt, sigma)

    loadRes = np.load(spikesRatesForParamsFilename)
    params = loadRes["params"]
    spikeRatesForParams = loadRes["spikeRates"]
    
    loadRes = np.load(ysFilename)
    ys = loadRes["ys"]
    sigma = loadRes["sigma"]
    trueParams = loadRes["params"]

    tf = min(tfSpikeRatesForParams, tfYs)
    nSamples = int(round(tf/dt))
    spikeRatesForParams = spikeRatesForParams[:, :nSamples, :]
    ys = ys[:nSamples,]
    llCalculator = LLCalculator()
    lls = llCalculator.calculateLL(ys=ys, spikeRatesForParams=
                                           spikeRatesForParams, sigma=sigma)
    np.savez(resultsFilenamePattern % (sigma), params=params, lls=lls)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

