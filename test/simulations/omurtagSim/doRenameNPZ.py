
import sys
import numpy as np
import math
import pdb
from IFSimulatorRWFWI import IFSimulatorRWFWI
from TwoPopulationsIFSimulator import TwoPopulationsIFSimulator

def main(argv):
    resultsFilename = "results/twoPSinusoidalRWFWINNeurons9000.npz"
    newResultsFilename = "results/twoPSinusoidalRWFWINNeurons9000b.npz"

    results = np.load(resultsFilename)
    np.savez(newResultsFilename,
             eTimes=results["edm1Times"],
             eSpikeRates=results["edm1SpikeRates"], 
             iTimes=results["edm2Times"], 
             iSpikeRates=results["edm2SpikeRates"],
             eEInputCurrentHist=results["edm1EInputCurrentHist"],
             eEFeedbackCurrentHist=results["edm1EFeedbackCurrentHist"],
             eIInputCurrentHist=results["edm1IInputCurrentHist"],
             eIFeedbackCurrentHist=results["edm1IFeedbackCurrentHist"],
             iEInputCurrentHist=results["edm2EInputCurrentHist"],
             iEFeedbackCurrentHist=results["edm2EFeedbackCurrentHist"],
             iIInputCurrentHist=results["edm2IInputCurrentHist"],
             iIFeedbackCurrentHist=results["edm2IFeedbackCurrentHist"])


if __name__ == "__main__":
    main(sys.argv)

