import pandas as pa
import numpy as np
import math
import glob
import csv
import os
from shapely.geometry import Polygon
from PIL import Image
from typing import Dict, List
from pathlib import Path
import json

class ComicImage:
    """
    Class that represent a ComicImage

    Attributes
        - id : the image number 
        - width, height : the width and the height of the image
        - filename : the full iamge name with extension
    """
    def __init__(self, imageNo, imageName, width, height, extension_length):
        self.imageNo = imageNo
        self.imageName = imageName
        self.imageNameShort = imageName[0:len(imageName)-extension_length]
        self.height = height
        self.width = width


    """
    Print the image characteristic
    """
    def toString(self):
        print("Image n°"+str(self.imageNo)+ " whose name is "+self.imageNameShort+ " and the dimension are ("+str(self.width)+"x"+str(self.height)+")")

class AOI:
    """
    Class that represent an AOI in a comic image

        - id : the id of the AOI in the image
        - image_id : the id of the corresponding image
        - category : the category of the AOI (text bubble, charachter,...)
        - segmentation : delimitation of the AOI
        - attributes : atttributes of the AOI
        - real_id: the id of the image in the json file
        - supercategory : super category of the aoi
        - area : area of the AOI
    """

    def __init__(self, aoi_id, image_id, category, segmentation, attributes, real_id, supercategory, area):
        self.aoi_id = aoi_id
        self.image_id = image_id
        self.category = category
        self.segmentation = segmentation
        self.attributes = attributes
        self.real_id = real_id
        self.supercategory = supercategory
        self.area = area
        
    def toString(self):
        print("AOI id "+str(self.aoi_id)+ ", whose image's id is "+ str(self.image_id) + " and the category is : " + str(self.category))

    def getCentroid(self, seg):
        """
        Return the center of a chosen segmentation of the AOI under the form of a Polygone.centroid
        seg : the segmentation we want to get the center of
        """
        polygon = Polygon([(seg[i], seg[i+1]) for i in range(0, len(seg), 2)])
        return polygon.centroid

def getComicImage(imageList,imageNo):
    """
    Return the comicImage object corresponding to the given image number
    arguments:
        - imageList : the list of all stimulis currently loaded
        - iamgeNo : the image id we want to find
    """
    for k in imageList:
        if(k.imageNo==imageNo):
            return k
    return None


def get_image_id(filename, data, conf):
    # Remove ".jpg1.borderless" from file name
    file_name = os.path.basename(filename)

    # Find the image ID in the data
    for img in data[0]["images"]:
        if (img["file_name"] == "yves/" + file_name or img["file_name"] == "placid/" + file_name):
            return img["id"]

    # Return -1 if no matching image is found
    return -1

        
def getImageList(conf):
    """
        This method will scan the configured STIMULI folder and create one instance of the ComicImage for each stimuli
        Return a list of all images 
    """
    imageList = []
    
    data = []
    
    with open(conf.AOIS_FILE, "r") as fh:
        for line in fh:
            data.append(json.loads(line))

    image_dir = conf.STIMULI

    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_id = get_image_id(filename, data, conf)
            if image_id == -1:
                continue

            for image in data[0]["images"]:
                if image["id"] == image_id:
                    imageList.append(ComicImage(image_id, image["file_name"].replace("yves/","").replace("placid/",""), image["width"], image["height"], conf.EXTENSION_LENGTH))

    return imageList
    
        
def getAOIListByImage(conf):
    """
    Method that read the configured json file to get every AOIs in each images
    This method return a dictionnary with all aois linked to their image id
    this method should be modified if only specific AOIs are needed 
    """
    
    aoisByImage = {}
    
    data = []

    with open(conf.AOIS_FILE, "r") as fh:
        for line in fh:
            data.append(json.loads(line))
        
    image_dir = conf.STIMULI

    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_id = get_image_id(filename, data, conf)
            if image_id == -1:
                continue

            AOIs = []
            count = 0
            for anno in data[0]["annotations"]:
                if anno["image_id"] == image_id:
         
                    #if(isinstance(anno["segmentation"], list)):
                    #if anno["category_id"] == 25 or anno["category_id"] == 24:
                    #If the annotation is horizon then don't count
                    if anno["category_id"] == 24:
                        continue
                      
                    if type(anno["segmentation"]) is dict:
                        print("Image " + str(anno["image_id"]) + " AOI " + str(anno["id"]) + str(anno["category_id"]))
                        continue
                        
                    if(len(anno["segmentation"]) != 0 and len(anno["segmentation"][0]) != 0):
                        seg = anno["segmentation"]
                    else:
                        continue
                    
                    supercategory = 0
                    for category in data[0]["categories"]:
                        if category["id"] == anno["category_id"]:
                            supercategory = category["supercategory"]
                            
                    
                        
                    AOIs.append(AOI(count, anno["image_id"], anno["category_id"], seg, anno["attributes"], anno["id"], supercategory, anno["area"]))
                    count += 1
                    
            
            AOIs.append(AOI(count, image_id, 21, [[]], [], 0, "BACKGROUND", 0))
            aoisByImage[image_id] = AOIs

    return aoisByImage


def computeCurrentDispThreshold(curSizePixel,conf):
    """
    Compute the dispersion threshold for the current image
    arguments
        - curSizePixel : the height of the current image.
    """
    deg_per_px = math.degrees(math.atan2(.5*conf.heightOnScreen, conf.distance_to_screen)) / (.5*curSizePixel)
    return conf.dispersion_thresold_degree/deg_per_px


"""
The implementation of this ID-T algorithm come from the github repository : https://github.com/ecekt/eyegaze, written by Ecekt
The implementation contains the idt and the get_dispersion methods
"""
def idt(data, conf, dis_threshold):

    window_range = [0,0]

    current = 0 #pointer to represent the current beginning point of the window
    last = 0

    #final lists for fixation info
    centroidsX = []
    centroidsY = []
    time0 = []
    time1 = []
    timestampFirstFix = np.max(data[:,conf.TIMESTAMP_INDEX])
    
    while (current < len(data)):
        
        t0 = float(data[current][conf.TIMESTAMP_INDEX]) #beginning time
        t1 = t0 + float(conf.duration_threshold)     #time after a min. fix. threshold has been observed

        for r in range(current, len(data)): 
            if(float(data[r][conf.TIMESTAMP_INDEX])>= t0 and float(data[r][conf.TIMESTAMP_INDEX])<=t1):
                #print "if",r
                last = r #this will find the last index still in the duration threshold

        window_range = [current,last]

        #now check the dispersion in this window
        #print "window", current, last
        dispersion = get_dispersion(data[current:last+1], conf)
        
        if (dispersion <= dis_threshold):
            #add new points to the fixation 
            while(dispersion <= dis_threshold and last + 1 < len(data)):

                last += 1
                window_range = [current,last]
                #print current, last, "*"
                #print "*"
                dispersion = get_dispersion(data[current:last+1], conf)
       
            #dispersion threshold is exceeded
            #fixation at the centroid [current,last]
            cX = 0
            cY = 0
            
            for f in range(current, last + 1):
                cX += float(data[f][conf.X_COORDINATES_INDEX])
                cY += float(data[f][conf.Y_COORDINATES_INDEX])

            cX = cX / float(last - current + 1)
            cY = cY / float(last - current + 1)
                
            t0 = data[current][conf.TIMESTAMP_INDEX]
            t1 = data[last][conf.TIMESTAMP_INDEX]
            timestampFirstFix = min(timestampFirstFix,t0)
            
            centroidsX.append(int(cX))
            centroidsY.append(int(cY))
            time0.append(t0)
            time1.append(t1)
            
            current = last + 1 #this will move the pointer to a novel window

        else:
            current += 1 #this will remove the first point -> Because with the minimal duration, it exceneeds the maximum trehsold distnace.
            last = current

    time0array = np.array(time0).astype(int)
    time1array = np.array(time1).astype(int)

    return np.stack((centroidsX, centroidsY, time0array, time1array -time0array, time0array-timestampFirstFix), axis=1)

def get_dispersion(points, conf):
    dispersion = 0
    
    argxmin = np.min(points[:, conf.X_COORDINATES_INDEX].astype(np.float))
    argxmax = np.max(points[:, conf.X_COORDINATES_INDEX].astype(np.float))
    
    argymin = np.min(points[:, conf.Y_COORDINATES_INDEX].astype(np.float))
    argymax = np.max(points[:, conf.Y_COORDINATES_INDEX].astype(np.float))

    dispersion = ((argxmax - argxmin) + (argymax - argymin))/2
    return dispersion

def getFixations(conf):
    """
    This function will read all the csv files in the conf.GAZEPOINTS folder. 
    Returns a dictionnary of this form : imageNo => List(participantNo, fixations)
    """
    fix = {}

    #Iterate over all files in the given folder
    for x in glob.glob(conf.GAZEPOINTS+"*.csv"):
        idx = -1
        participantId = 0
        curImg = None
        
        #Get the image number of the current CSV File
        for i in conf.imageList:
            
            if i.imageNameShort in x:
                idx = i.imageNo
                curImg = i
                participantId = x[len(conf.GAZEPOINTS+i.imageNameShort+".jpg_participant_"):len(x)-4]
                             
        #If there is no stimuli, discard the file (Probably a file used to test the accuracy of the material)
        if(idx!=-1):
            #Get all gazepoints into a numpy array
            with open(x, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                headers = next(reader) #Just to remove the header line in the CSV
                data = np.array(list(reader)).astype(np.int)
               
            
            #If there is at least one gazepoint
            if (data.size):

                #Use the idt algorithm to transform gazepoints into fixations (if configured)
                if(conf.useIDT):
                    curDispThreshold = computeCurrentDispThreshold(curImg.height,conf)
                    datas = idt(data, conf, curDispThreshold)
                else:
                    #Add two columns to fixation table
                    datas = np.concatenate((data,np.zeros((data.shape[0],2),dtype=np.int)),axis=1)

                    #Compute the fixation time and the time spent from the first fixation of the file
                    for i in range (0,datas.shape[0]-1):
                        datas[i,3]=datas[i+1,2]-datas[i,2]
                        datas[i,4] = datas[i,2]-datas[0,2]

                if(datas.any()):
                    #Remove the last line
                    array = datas[0:(datas.shape[0]-1),0:datas.shape[1]]

                    #Add the fixation table to the dictionnary
                    if(array.any()):
                        fix.setdefault(idx,list()).append((participantId,array))
    return fix