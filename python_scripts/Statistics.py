import numpy as np
import LoadDatas as ld
import Visualisations as vs
from matplotlib.path import Path
from PIL import Image, ImageDraw, ImageFont
import math
import cv2
import Config as conf

class AOIStatistic:
    """
    Class that represent the statistics for one AOI 

    Attributes : 
        - viewingTime: the total time of fixation in this AOI
        - timeToFirstFixation: Time until the first fixation in the AOI
        - gazeNumber: The number of gazes in this AOI
        - fixationNumber : The fixation count in this AOI
    """
    def __init__(self, viewingTime, timeToFirstFixation, gazeNumber , fixationNumber):
        self.viewingTime = viewingTime
        self.timeToFirstFixation = timeToFirstFixation
        self.gazeNumber = gazeNumber
        self.fixationNumber = fixationNumber
    
    """
    Add one fixation to the current AOIs
    arguments: 
        fixDuration: The current fixation's duration
        incGazeNumber: If we need to add one more gaze in the AOI statistics
        timeToFirstFixation: the number of millisecond after which this fixation comes
        fixationNumber: the number of fixation we add (set to 0 if we need to only add the final gaze)
    """
    def update(self, fixDuration, incGazeNumber=False, timeToFirstFixation=0, fixationNumber = 1):
        #Test if it is the first time that we update this AOI statistics
        if(self.timeToFirstFixation==-1):
            self.timeToFirstFixation=timeToFirstFixation
        
        self.fixationNumber += fixationNumber
        self.viewingTime += fixDuration
        if incGazeNumber:
            self.gazeNumber += 1
            
                       
class UserStatisticsByImage:
    """
    Class that represent the statistic for one user for one image.

    Attributes: 
        - imageNo : the image number for which the user statistic is computed
        - userId : the user for which  the statistics are computed
        - nbAoisInImage: the number of aois in the image (to create all AOIStatics objects)
        - totalDuration: the total duration of fixation in the current image
        - totalNbFixation: the total number of fixation in the current image
    """
    def __init__(self, imageNo, userId, nbAoisInImage):
        self.imageNo = imageNo
        self.userId = userId
        self.aoiStats = {}
        
        for i in range(nbAoisInImage):
            self.aoiStats[i]= AOIStatistic(0, -1, 0, 0)
            
        self.totalDuration = 0
        self.totalNbFixation = 0
        
    
    def update(self, aoiNumber, fixDuration, timeToFirstFixation = 0, incGazeNumber = False, fixationNumber = 1):
        """
        Update the statistic by adding one fixation into it
        arguments: 
            - fixDuration : the duration of the fixation
            - aoiNumber : the aoi statistic we want to update

        optional:
            - timeToFirstFixation: the timestamp from the first fixation until the current one
            - incGazeNumber: If we need to increment the gaze number for the current AOI
            - fixationNumber: the number of fixation we add (set to 0 if we need to only add the final gaze)
        """
        aoistat = self.aoiStats[aoiNumber]
        #print(aoistat)
        
        aoistat.update(fixDuration, incGazeNumber, timeToFirstFixation,fixationNumber)
            
        self.totalNbFixation += fixationNumber
        self.totalDuration += fixDuration

    
    """
    Return the AOI statistics for the asked aoi number
    arguments: 
        - aoiNumber : The aoi number
    """
    def getAOIStatistics(self, aoiNumber):
        return self.aoiStats[aoiNumber]
    
    """
    Print the statistics for the current users
    """
    def printStatistics(self):
        for aoi in sorted(self.aoiStats):
            curAoi = self.aoiStats[aoi]
            print("=======================================================================================")
            print("Statistics of user " + str(self.userId) + " in image " +str(self.imageNo)+ " for AOI number " + str(aoi))
            print("=======================================================================================")
            print("\tNumber of fixation : "+ str(curAoi.fixationNumber))
            print("\tFixation durations (ms) : "+ str(curAoi.viewingTime))
            if(curAoi.fixationNumber!=0):
                print("\tMean fixation duration for the user (ms) : "+ str(round(curAoi.viewingTime/curAoi.fixationNumber, 2)))
            print("\tNumber of Gazes : " + str(curAoi.gazeNumber))
            print("\tTime to first fixation (ms) : " + str(curAoi.timeToFirstFixation))
            print("\tPercentage of fixations number on this AOI: "+ str(round(curAoi.fixationNumber/self.totalNbFixation*100, 2)))
            print("\tPercentage of fixation duration on this AOI: "+ str(round(curAoi.viewingTime/self.totalDuration*100, 2)))
            
class ImageStatistics:
    """
    Class that represents that represents the statistics among all users for one image

    Attributes:
        - imageNo : The current image id
        - userStatistics: a list containing all user statistics for the current image
    """
    def __init__(self, imageNo):
        self.imageNo = imageNo
        self.userStatistics = []
    
    def addAOIStatistics(self, userStat):
        """
        Add the given user statistic to the list of user statistics
        Arguments:
            - userStat : the user statistic that need to be addded
        """
        self.userStatistics.append(userStat)

    def computeImageStatistics(self, aoiNbs, conf):
        """
        Merge all the users statistics and compute the global statistics for this image. The value returned will be a dictionnary of the form  
                metrics => {
                    mean => 
                    stdDev =>
                    boundUp=>
                    bounDown => 
                }
        arguments: 
            - aoiNbs : the number of AOIs in the current image
            - conf: the config object
        """
        statByAOI = {}
        
        for i in range(aoiNbs):
            #For each AOI, create empty lists for all metrics
            fixNumberList = []
            gazeNumberList = []
            fixationDurationList = []
            timeToFirstFixationList = [] 
            viewTimeList = []

            #Intialised basic values for the range. 
            minFixNumber = -1
            maxFixNumber = -1
            minFixDur = -1
            maxFixDur = -1
            minGazNum = -1
            maxGazNum = -1
            minTtff = -1
            maxTtff = -1
            minViewTime = -1
            maxViewTime = -1
            
            #Iterate overall user statistics for the current AOI to merge them
            for curUserStat in self.userStatistics:
                curAOI = curUserStat.getAOIStatistics(i)

                #Get the value for the current user statistic
                fixNumber = curAOI.fixationNumber
                viewTime = curAOI.viewingTime*1.0/1000
                fixDur = round(0 if fixNumber == 0 else curAOI.viewingTime/fixNumber,3)
                GazNum = curAOI.gazeNumber
                ttff = curAOI.timeToFirstFixation
                
                #Update the lists
                fixNumberList.append(fixNumber)
                gazeNumberList.append(GazNum)
                fixationDurationList.append(fixDur)
                viewTimeList.append(viewTime)
                
                #the user did see this AOI, we should not append the time to first fixation as it is -1
                if(ttff!=-1):
                    timeToFirstFixationList.append(ttff)

                #Update the ranges
                if(maxFixNumber ==-1):
                    minFixNumber = fixNumber
                    maxFixNumber = fixNumber
                    minFixDur = fixDur
                    maxFixDur = fixDur
                    minGazNum = GazNum
                    maxGazNum = GazNum
                    minTtff = ttff
                    maxTtff = ttff
                    minViewTime = viewTime
                    maxViewTime = viewTime
                    
                else:
                    minFixNumber = min(fixNumber,minFixNumber)
                    maxFixNumber = max(fixNumber,maxFixNumber)
                    minFixDur = min(fixDur, minFixDur)
                    maxFixDur = max(fixDur, maxFixDur)
                    minGazNum = min(GazNum, minGazNum)
                    maxGazNum = max(GazNum, maxGazNum)
                    minViewTime = min(viewTime, minViewTime)
                    maxViewTime = max(viewTime, maxViewTime)
                    
                    #If the user did see this AOI
                    if(ttff!=-1):
                        minTtff = min(ttff, minTtff)
                        maxTtff = max(ttff, maxTtff)

            #Transform list into array and use numpy to compute the mean and the standard deviation for each metric
            arrayFixNb = np.array(fixNumberList)
            meanFixNb = round(np.mean(arrayFixNb),3)
            stdDevFixNb = round(np.std(arrayFixNb),3)

            arrayViewTime = np.array(viewTimeList)
            meanViewTime = round(np.mean(arrayViewTime),3)
            stdViewTime = round(np.std(arrayViewTime),3)
            
            arrayFixDur = np.array(fixationDurationList)
            meanFixDur = round(np.mean(arrayFixDur),3)
            stdDevFixDur = round(np.std(arrayFixDur),3)

            arrayGazNb = np.array(gazeNumberList)
            meanGazNb = round(np.mean(arrayGazNb),3)
            stdDevGazNb = round(np.std(arrayGazNb),3)

            arrayTime = np.array(timeToFirstFixationList)
            meanTTFF = round(np.mean(arrayTime),3)
            stdDevTTFF = round(np.std(arrayTime),3)

            #Store the statistics in the dictionnary
            statByAOI[i] = {"number": len(fixNumberList),
                            conf.categories[4]: {"mean": meanViewTime, "stdDev": stdViewTime, "boundUp": maxViewTime, "boundDown": minViewTime},
                            conf.categories[0]: {"mean": meanFixNb, "stdDev": stdDevFixNb, "boundUp": maxFixNumber, "boundDown": minFixNumber},
                            conf.categories[1]: {"mean": meanFixDur, "stdDev": stdDevFixDur, "boundUp": maxFixDur, "boundDown": minFixDur},
                            conf.categories[2]: {"mean": meanGazNb, "stdDev": stdDevGazNb, "boundUp": maxGazNum, "boundDown": minGazNum},
                            conf.categories[3]: {"mean": meanTTFF, "stdDev": stdDevTTFF, "boundUp": maxTtff, "boundDown": minTtff}}

        return statByAOI

def getAOINumberInImage(aoisByImage, imageNo, x, y):
    """
    Given an image and given a point x,y in the image,
    this function returns the annotation id if x,y is in a certain annotation and -1 otherwise.

    Parameters:
        aoisByImage (Dict[int, List[AOI]]): a dictionary linking image numbers to lists of AOIs
        imageNo (int): the number of the image to look for AOIs in
        x (int): the x-coordinate of the point to check
        y (int): the y-coordinate of the point to check

    Returns:
        int: the id of the AOI containing the point (x, y), or -1 if no AOI contains the point
    """

    if imageNo not in aoisByImage:
        return -1
    
    aois = aoisByImage[imageNo]

    possibleAOIs = []
    
    for aoi in aois:
           
        seg_tuples = [tuple()]
        
        for seg in aoi.segmentation:
            if len(seg) == 0:
                continue 
                
            for i in range(len(seg) - 1):
                if(i % 2 == 0):
                    seg_tuples.append((seg[i], seg[i+1]))
        
            seg_tuples = seg_tuples[1:]
        

        # Create a Path object from the vertices

            path = Path(np.array(seg_tuples))
            if path.contains_point((x, y)):
                possibleAOIs.append(aoi)
    
    if(len(possibleAOIs) == 0):
        return aoisByImage[imageNo][len(aoisByImage[imageNo]) - 1].aoi_id
    
    previousSuper = ""
    newId = -1
    if len(possibleAOIs) > 1:
        for aoi in possibleAOIs:
            if(aoi.supercategory == "TEXT"):
                return aoi.aoi_id
            elif(aoi.supercategory == "CHARACTER"):
                previousSuper = "CHARACTER"
                newId = aoi.aoi_id
            elif(aoi.supercategory == "ANIMAL" and previousSuper != "CHARACTER"):
                previousSuper = "ANIMAL"
                newId = aoi.aoi_id
            elif(aoi.supercategory == "OBJECT" and previousSuper != "CHARACTER" and previousSuper != "ANIMAL"):
                previousSuper = "OBJECT"
                newId = aoi.aoi_id
            elif(aoi.supercategory == "BACKGROUND" and previousSuper != "CHARACTER" and previousSuper != "ANIMAL" and previousSuper != "OBJECT"):
                previousSuper = "BACKGROUND"
                newId = aoi.aoi_id
                
        return newId
                
    return possibleAOIs[0].aoi_id


def changeFixations(conf):
    """
    Changes the fixations dictionary for the heatmaps generation

    Returns:
        dictionary: the update fixations dictionary based on the param text
    """
    
    fixationsNoText = {}
    fixationsOnlyText = {}
    
    for k in conf.fixations:
        for tup in conf.fixations[k]:
            arrayOnlyText = np.empty((0, tup[1].shape[1]), dtype=tup[1].dtype)
            arrayNoText = np.empty((0, tup[1].shape[1]), dtype=tup[1].dtype)
            
            #Iterate over all fixations of the currentUser in the image k
            
            for fix in tup[1]:
                aoiNb = getAOINumberInImage(conf.aoisByImage,k,fix[conf.X_COORDINATES_INDEX],fix[conf.Y_COORDINATES_INDEX])
                
                #If the point is not in an AOI keep the fixation
                if(aoiNb == -1):
                    arrayNoText = np.vstack([arrayNoText, fix])
                    continue
                
                #Gets the category of the aoi on which the point is
                category = conf.aoisByImage[k][aoiNb].category
                
                #If the point is on text, keep fixation for onlyText
                if(category == 27 or category == 26):
                    arrayOnlyText = np.vstack([arrayOnlyText, fix])
                    
                #If the point is on an AOI that's not text or comic bubble
                else:
                    arrayNoText = np.vstack([arrayNoText, fix])
                
            fixationsNoText.setdefault(k,list()).append((tup[0],arrayNoText))
            fixationsOnlyText.setdefault(k,list()).append((tup[0],arrayOnlyText))
            
            
    return (fixationsNoText, fixationsOnlyText)
                        


def getUserStatistics(conf):
    """
    Compute the statistics by user for each image 
    Returns a dictionnary of the form 
        participantNo => { imageNo => UserStatisticsByImage} 
    """
    statByUser={}
    #print('this is it',conf.fixations.keys())
    #Iterate over all Images
    for k in conf.fixations:
        #iterate over all (userId, fixations) for the currentImage
        
        
        #print('k:',k)
        #print('conf.aoisByImage' ,conf.aoisByImage)
        
        for tup in conf.fixations[k]:
            #print('tup:',tup)
            lastAOINb = -1
            userByImageStat = UserStatisticsByImage(k,tup[0],len(conf.aoisByImage[k])) 
            
            #Iterate over all fixation of the currentUser in the image k
            for fix in tup[1]:
                aoiNb = getAOINumberInImage(conf.aoisByImage,k,fix[conf.X_COORDINATES_INDEX],fix[conf.Y_COORDINATES_INDEX])
                
                #If the fixation is not in an AOI, we discard it 
                if(aoiNb!=-1):
                    #If the aoi number of the preivous and the current one are not the same, we ened to update the gaze number. 
                    incGaze = lastAOINb != aoiNb and lastAOINb != -1
                    
                    userByImageStat.update(aoiNb, fix[conf.FIXDURATION_INDEX], timeToFirstFixation = fix[conf.TIMEFROMSTART_INDEX], incGazeNumber = incGaze)
                    lastAOINb = aoiNb

            if(aoiNb!=-1):
                userByImageStat.update(aoiNb, 0,0,True,0)
            
            dic = {k: userByImageStat}
            #If the user is already in the dictionnary
            if tup[0] in statByUser:
                dic = statByUser[tup[0]]
                dic[k] = userByImageStat
            
            statByUser[tup[0]] = dic
    return statByUser 

def computeGlobalStatistic(statByUser, conf):
    """
    Combine the user statistics to create statistics by image.
    Returns a dictionnary of the form 
        imageNo => {
                metrics => {
                    mean => 
                    stdDev =>
                    boundUp=>
                    bounDown => 
                } 
            }
    argument 
        - statByUser : All the user statistics 
        - conf : the config object
    """
    statByImages={}
    
    #Iterate over all user statistics and add them in the correct ImageStatistics object
    for userId in statByUser:
        dicImagesCurUser = statByUser[userId]
        for imId in dicImagesCurUser:
            if imId in statByImages:
                imageStat = statByImages[imId]
                imageStat.addAOIStatistics(dicImagesCurUser[imId])
            else:
                imageStat = ImageStatistics(imId)
                imageStat.addAOIStatistics(dicImagesCurUser[imId])
                statByImages[imId] = imageStat
    
    #Compute statistics by images 
    finalStatByImages = {}
    for imId in statByImages:
        finalStatByImages[imId] = statByImages[imId].computeImageStatistics(len(conf.aoisByImage[imId]),conf)
    
    return finalStatByImages

def printStatistics(statistics, conf):
    """
    Print the global statistics
    arguments:
        - statistics: dictionnary of the form imageNo => {
                                                    metrics => {
                                                        mean => 
                                                        stdDev =>
                                                        boundUp=>
                                                        bounDown => 
                                                    } 
                                                }
        - conf: the config object
    """
    print("================================")
    print("Average statistics over all users")
    print("================================")
    for i in sorted(statistics.keys()):
        print("Image n°" + str(i) + " ("+ld.getComicImage(conf.imageList,i).imageNameShort+")")
        statByImages = statistics[i]
        for aoi in sorted(statByImages.keys()):
            aoiDic = statByImages[aoi]
            fixNb = aoiDic[conf.categories[0]]
            fixDur = aoiDic[conf.categories[1]]
            gazNumber = aoiDic[conf.categories[2]]
            ttff = aoiDic[conf.categories[3]]
            viewTime = aoiDic[conf.categories[4]]
            
            print("\t----------------")
            print("\tAOI n°"+str(aoi) + " (participation of " +str(aoiDic["number"])+" observers)")
            print("\t\tMean fixation number : " + str(fixNb["mean"]) + " (SD = " + str(fixNb["stdDev"]) + ") | Range : " + str(fixNb["boundDown"]) + " - "+str(fixNb["boundUp"]))
            print("\t\tMean viewing time (sec) : " + str(viewTime["mean"]) + " (SD = " + str(viewTime["stdDev"]) + ") | Range : " + str(viewTime["boundDown"]) + " - "+str(viewTime["boundUp"]))
            print("\t\tMean fixation duration (ms) : " + str(fixDur["mean"]) + " (SD = " + str(fixDur["stdDev"]) + ") | Range : " + str(fixDur["boundDown"]) + " - "+str(fixDur["boundUp"]))
            print("\t\tMean gaze number : " + str(gazNumber["mean"]) + " (SD = " + str(gazNumber["stdDev"]) + ") | Range : " + str(gazNumber["boundDown"]) + " - "+str(gazNumber["boundUp"]))
            print("\t\tMean time to first fixation (ms) : " + str(ttff["mean"]) + " (SD = " + str(ttff["stdDev"]) + ") | Range : " + str(ttff["boundDown"]) + " - "+str(ttff["boundUp"]))
            print("\t----------------")

