
import sys
import pickle
import pdb
import numpy as np

def getExtremeCurrentsForEDM(results, eInputFilename, eFeedbackFilename, 
                                      iInputFilename, iFeedbackFilename):
    maxEStimForEDM = minEStimForEDM = maxIStimForEDM = minIStimForEDM = None
    eStimForEDM = iStimForEDM = None

    if eInputFilename in results.keys() and \
       eFeedbackFilename in results.keys():
        eStimForEDM = results[eInputFilename] + results[eFeedbackFilename]
    elif eInputFilename in results.keys() and \
         not eFeedbackFilename in results.keys():
        eStimForEDM = results[eInputFilename]
    elif eFeedbackFilename in results.keys(): 
        eStimForEDM =  results[eFeedbackFilename]

    if eStimForEDM is not None:
        maxEStimForEDM = eStimForEDM.max()
        minEStimForEDM = eStimForEDM.min()

    if iInputFilename in results.keys() and \
       iFeedbackFilename in results.keys():
        iStimForEDM = results[iInputFilename] + results[iFeedbackFilename]
    elif iInputFilename in results.keys() and \
         not iFeedbackFilename in results.keys():
        iStimForEDM = results[iInputFilename]
    elif iFeedbackFilename in results.keys(): 
        iStimForEDM =  results[iFeedbackFilename]

    if iStimForEDM is not None:
        maxIStimForEDM = iStimForEDM.max()
        minIStimForEDM = iStimForEDM.min()

    return(maxEStimForEDM, minEStimForEDM, maxIStimForEDM, minIStimForEDM)

def main(argv):

    if(len(argv)!=2):
        print("Usage: %s <intetrationResult.npz>" % argv[0])
        sys.exit()

    eInputNamePattern = "edm%dEInputCurrentHist"
    iInputNamePattern = "edm%dIInputCurrentHist"
    eFeedbackNamePattern = "edm%dEFeedbackCurrentHist"
    iFeedbackNamePattern = "edm%dIFeedbackCurrentHist"

    results = np.load(argv[1])
    maxEStim = -float('inf')
    minEStim = float('inf')
    maxIStim = -float('inf')
    minIStim = float('inf')

    edmIndex = 1
    exit = False

    while(not exit):
        eInputFilename = eInputNamePattern % edmIndex
        iInputFilename = iInputNamePattern % edmIndex
        eFeedbackFilename = eFeedbackNamePattern % edmIndex
        iFeedbackFilename = iFeedbackNamePattern % edmIndex

        maxEStimForEDM, minEStimForEDM, maxIStimForEDM, minIStimForEDM = \
         getExtremeCurrentsForEDM(results=results, 
                                   eInputFilename=eInputFilename, 
                                   eFeedbackFilename=eFeedbackFilename, 
                                   iInputFilename=iInputFilename, 
                                   iFeedbackFilename=iFeedbackFilename)
        if maxEStimForEDM is None and \
           minEStimForEDM is None and \
           maxIStimForEDM is None and \
           minIStimForEDM is None:
            exit = True
        else:
            if maxEStimForEDM is not None and maxEStimForEDM>maxEStim:
                maxEStim = maxEStimForEDM
            if minEStimForEDM is not None and minEStimForEDM<minEStim:
                minEStim = minEStimForEDM
            if maxIStimForEDM is not None and maxIStimForEDM>maxIStim:
                maxIStim = maxIStimForEDM
            if minIStimForEDM is not None and minIStimForEDM<minIStim:
                minIStim = minIStimForEDM
        edmIndex = edmIndex + 1

    print(r"EStim $\in$ (%f, %f)" % (minEStim, maxEStim))
    print(r"IStim $\in$ (%f, %f)" % (minIStim, maxIStim))
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)

