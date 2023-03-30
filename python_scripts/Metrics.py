import numpy as np
import Visualisations as vs

"""
This implementation of AUC Judd and normalize_map comes from the GitHub repository : https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py
the implementation contains the two following methods
"""
def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map))*1.0)
    return norm_s_map

def auc_judd(s_map, gt, plotCurve=False):
    # ground truth is discrete, s_map is continous and normalized
    s_map = normalize_map(s_map)
    assert np.max(gt) == 1.0,\
        'Ground truth not discretized properly max value > 1.0'
    assert np.max(s_map) == 1.0,\
        'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map[gt > 0].tolist()

    num_fixations = len(thresholds)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map,
        # keep only those pixels with values above threshold
        temp = s_map >= thresh
        num_overlap = np.sum(np.logical_and(temp, gt))
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[:2]) - num_fixations)

        area.append((round(tp, 4) ,round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list, fp_list = list(zip(*area))
    if(plotCurve):
        vs.plotROCCurve(fp_list,tp_list)

    return np.trapz(np.array(tp_list), np.array(fp_list))
    
"""
The following two methods come from the Github repository: https://github.com/dariozanca/FixaTons/blob/master/FixaTons/_visual_attention_metrics.py
"""

''' created: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017
This finds the normalized scanpath saliency (NSS) between two different saliency maps. 
NSS is the average of the response values at human eye positions in a model saliency 
map that has been normalized to have zero mean and unit standard deviation. '''

def NSS(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)


    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    if not saliencyMap.max() == 0:
        map1 = saliencyMap.astype(float) / saliencyMap.max()

    # normalize saliency map
    if not map1.std(ddof=1) == 0:
        map1 = (map1 - map1.mean()) / map1.std(ddof=1) 

    # mean value at fixation locations
    score = map1[fixationMap.astype(bool)].mean()

    return score


######################################################################################

''' created: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017
This finds the KL-divergence between two different saliency maps when viewed as 
distributions: it is a non-symmetric measure of the information lost when saliencyMap 
is used to estimate fixationMap. '''


def KLdiv(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map

    # convert to float
    map1 = saliencyMap.astype(float)
    map2 = fixationMap.astype(float)


    # make sure map1 and map2 sum to 1
    if map1.any():
        map1 = map1 / map1.sum()
    if map2.any():
        map2 = map2 / map2.sum()

    # compute KL-divergence
    eps = 10 ** -12
    score = map2 * np.log(eps + map2 / (map1 + eps))

    return score.sum()

  
def computeSumBinMapsByImages(binMapDic, conf):
    """
    Sum all binary maps that corresponds to the same image
    arguments
        - binMapDic: The dictionnary containing all the binary maps in the following format : (imageNo => List(participantId, binaryMap))
        - conf : The config object
    """
    sumBinMapsByImage = {}
    for imageNo in binMapDic:
        #Create a new array with the correct shape
        current = np.zeros(((binMapDic[imageNo])[0])[1].shape)

        for tup in binMapDic[imageNo]:
            current += tup[1]

        sumBinMapsByImage[imageNo] = current
    return sumBinMapsByImage

"""
The input should be a dictionnary of the form (key => v) = 
"""
def computeIOCScore(binMapDic, sumBinMapsByImage, conf, plotCurve=False, saveSaliencyMap=False, saveCombinedMaps = False):
    """
    Returns the IOC scores for each participant by images and the mean IOC scores by images. i.e. the output is of the form 
        imageNo: List(participantNo, {AUC=> score, KL=> score, NSS=>Score}), imageNo : {AUC=>(meanScore, std), KL=>(meanScore, std), NSS=>(meanScore, std)} 
    arguments
        - binMapDic: The dictionnary containing all the binary maps in the following format : (imageNo => List(participantId, binaryMap))
        - sumBinMapsByImage: A dictionnary of the form (imageNo => sumBinMaps) where sumBinMaps denotes the sum of every binary map for this image
        - conf : The conf object
    Optional
        - plotCurve: If we want to display the ROC Curve in Juypter notebbok
        - saveSaliencyMap : If we want to save the saliency maps in the configured folder. 
        - saveCombinedMaps: If we want to save the saveCombinedMaps (total binary map - current user binary map) in the configured folder.
    """
    IOCScore = {}
    IOCScoreMean = {}

    #Iterate over all images 
    for imageNo in binMapDic:
        curList = []
        
        curMeanAUC = []
        curMeanKL = []
        curMeanNSS = []

        curImg = conf.imageList[imageNo]
        totalBinMap = sumBinMapsByImage[imageNo]

        #Iterate over all saliency map, i.e. each observer of this 
        for tup in binMapDic[imageNo]:
            current = tup[0]
            salmapSolo = vs.computeSaliencyMap(np.copy(tup[1]),imNo = imageNo, saveFiles = saveSaliencyMap, partNo=current, conf=conf)

            gt = np.zeros(salmapSolo.shape).astype(int)
            gt[(totalBinMap-tup[1])>=1]=1

            if saveCombinedMaps:
                vs.saveMap(gt, imageNo, tup[0], conf, combBinMap=True)

            #Compute AUC score and store it on the list 
            curAuc = auc_judd(np.copy(salmapSolo), gt, plotCurve=plotCurve)
            curKL = KLdiv(np.copy(salmapSolo), gt)
            curNSS = NSS(np.copy(salmapSolo), gt)

            curMeanAUC.append(curAuc)
            curMeanKL.append(curKL)
            curMeanNSS.append(curNSS)
            curList.append((current, {"AUC":curAuc, "KL":curKL, "NSS":curNSS}))

        #Store the list in the dictionnary
        IOCScore[imageNo] = curList
        arrayAUC = np.array(curMeanAUC)
        arrayKL = np.array(curMeanKL)
        arrayNSS = np.array(curMeanNSS)

        IOCScoreMean[imageNo] = {"AUC": (np.mean(arrayAUC),np.std(arrayAUC)), "KL":(np.mean(arrayKL),np.std(arrayKL)), "NSS":(np.mean(arrayNSS), np.std(arrayNSS))}

    return IOCScore, IOCScoreMean

def printIOCScores(IOCScore, IOCScoreMean, conf):
    """
    Print the IOC scores and IOC score mean
    arguments:
        - IOCScore : dictionnary of the format: imageNo => List(participantNo, score) containing all IOC scores
        - IOCScoreMean: dictionnary of the format: imageNo => number containing IOC score means by images.
        - conf: The config object
    """
    for imageNo in sorted(IOCScore.keys()):
        print("====================================================")
        print("IOC Scores for image n°"+str(imageNo)+ " ("+str(conf.imageList[imageNo].imageNameShort)+")")
        print("\tMean score for the current image : ")
        curMeanScore = IOCScoreMean[imageNo]
        for k in curMeanScore:
            print("\t\t"+k+" score : " +str(round((curMeanScore[k])[0],3))+" (STD : "+str(round((curMeanScore[k])[1],3))+")")

        for tup in IOCScore[imageNo]:
            print("\tParticipant "+str(tup[0])+" gets the following scores: ")
            scores = tup[1]
            for j in scores:
                print("\t\t"+j+" score : "+str(round(scores[j],3)))
        print("====================================================")

