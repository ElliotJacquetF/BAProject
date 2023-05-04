import numpy as np
from matplotlib import image, pyplot, lines, font_manager as fm
from PIL import Image, ImageDraw, ImageFont
import os
import Statistics as st
import LoadDatas as ld
import math
import cv2
from shapely.geometry import Polygon


class Gaze:
    """
    This class represent a gaze witht the following: 
        - centerX : the x coordinates of the center gravity depending on fixation count
        - centerY : the y coordinates of the center gravity depending on fixation count
        - centerDurX : the x coordinates of the center gravity depending on fixation duration
        - centerDurY : the y coordinates of the center gravity depending on fixation duration
        - fixNb : Number of fixations inside this gaze
        - fixDuration : total duration of the fixations inside the gaze
    """
    def __init__(self, x, y, fixDuration):
        self.sumFixCountX = x
        self.sumFixCountY = y
        self.sumFixDurX = x*fixDuration
        self.sumFixDurY = y*fixDuration
        self.fixNb = 1
        self.fixDuration = fixDuration
    
    """
    Update the gazes by adding a new fixation into it
    """
    def update(self, fixX, fixY, fixDuration):
        self.sumFixCountX += fixX
        self.sumFixCountY += fixY
        self.sumFixDurX += fixX*fixDuration
        self.sumFixDurY += fixY*fixDuration

        self.fixNb +=1
        self.fixDuration += fixDuration

    """
    Compute gaze center
    """
    def computeCenter(self):
        self.centerCountX = self.sumFixCountX*1.0/self.fixNb
        self.centerCountY = self.sumFixCountY*1.0/self.fixNb
        self.centerDurX = self.sumFixDurX*1.0/self.fixDuration
        self.centerDurY = self.sumFixDurY*1.0/self.fixDuration
    
    """
    Print the gaze characteristics
    """
    def toString(self):
        print(f"Gaze centered in ({self.centerX},{self.centerY}) with {self.fixNb} fixations inside and a total duration {self.fixDuration}")


def drawAOIOverImages(conf):
    """
    This method represent graphically the AOI on images and save them in the configured folder.
    """

    for i in conf.imageList:
        #Create Image object
        im = Image.open(conf.STIMULI+i.imageName)
        idx = i.imageNo
        font_path = os.path.join("fonts", "Arial.ttf")

        
        #Draw Image
        draw = ImageDraw.Draw(im)
        
        
        #Draw rectangle that represents the AOI with the AOI id too. 
        for aoi in conf.aoisByImage[idx]:

            if (len(aoi.segmentation) == 0):
                continue
            
            for seg in aoi.segmentation:

                if len(seg) == 0:
                    continue
                    
                segmentations = seg
                
                # Draw the polygon on the image for each annotation
                #for seg in segmentations:
                draw.polygon(segmentations, outline ="red")
                
                
                # Define the font to use for the text
                font = ImageFont.truetype(font_path, size=40)

                # Compute the centroid of the polygon
                centroid = aoi.getCentroid(seg)
            
                text_width, text_height = draw.textsize(str(aoi.aoi_id), font=font)

                # Calculate the coordinates to center the text
                text_x = int(centroid.x) - text_width // 2
                text_y = int(centroid.y) - text_height // 2
            
                # Draw the text on the image
                draw.text((text_x, text_y), str(aoi.aoi_id), fill='black', font=font)
                        
        
        #Save image on the configured folder
        im.save(conf.AOI_FOLDER+i.imageName, "PNG")

"""
Method used to create a basic figure
    The returned figure will contains the image in background and axis at the correct scale but inverted 
"""        
def genFigure(figsize, dpi, dispsize):
    """
    Returns one matplotlib.pyplot figures and its axes, with a size of
    dispsize
    arguments 
        - figsize: The figure size in inch
        - dpi : the figure's DPI
        - dispsize:  Figure's size in pixels tuple(width, height)
    """
    #Create figure according to the given parameters
    fig = pyplot.figure(figsize=figsize, dpi = dpi, frameon = False)
    
    #Axes for figure
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    #Set the limit of the axis 
    ax.axis([0, dispsize[0], 0, dispsize[1]]) 
    return fig, ax


"""
Method that generate multiple figure for a given image number 
The return values will be figure1, axis for figure 1, figure 2, axis for figure 2, image name and the display size of the images. 

The methods getBasicFigures, genFigure,  gaussian, drawHeatmap and computeHeatmap are based on this implementation of the heatmap : https://github.com/TobiasRoeddiger/GazePointHeatMap
"""
def getBasicFigures(imageNo, conf, drawImage = True, alpha=1.0):
    """
    Returns two matplotlib.pyplot figures and their axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it (if asked) and also the image name
    arguments
        - imageNo: that need to be on the figure
        - conf : the config object
    returns
    optional
        - drawImage : If the image need to be draw on the figure or not
        - alpha : the opacity of the image on the figure. 
    """
    #Get the image 
    comicImage = ld.getComicImage(conf.imageList,imageNo)
    imgName = comicImage.imageName
    
    #Get image width and height
    dispsize = (comicImage.width, comicImage.height)


    screen = np.zeros((dispsize[1], dispsize[0], 3), dtype='float32')
    
    # load image
    img = image.imread(conf.STIMULI + imgName)

    # width and height of the image
    w, h = len(img[0]), len(img)
    
    # x and y position of the image on the display
    x = 0
    y = 0

    # draw the image on the screen
    if drawImage:
        screen[y:y + h, x:x + w, :] += img
        
    # dots per inch
    dpi = 100.0
    
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    
    # create a figure
    fig1, ax1 = genFigure(figsize, dpi, dispsize)
    fig2, ax2 = genFigure(figsize, dpi, dispsize)
    
    ax1.imshow(screen,alpha=alpha)  
    ax2.imshow(screen,alpha=alpha)

    return fig1, ax1, fig2, ax2, imgName

def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution
    arguments
    	- x : width in pixels
		- sx : width standard deviation
    keyword argments
		- y : height in pixels (default = x)
		- sy : height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))
    return M


def drawHeatmap(heatmap, conf, savefilename, ax, fig, saveFiles=True, alpha=0.5):
    """
    Draw the heatmap over the images and eventually save it.
    arguments:
        - ax, fig : should contain a figure with the stimuli already diplayed
        - savefilename : the file which where we need to save the heatmap 
        - conf: the config object
        - heatmap: numpy array containing the heatmap values
    optional:
        - saveFiles: boolean, if we want to save file or just plot them 
        - alpha: The degree of transparency of the heatmap over the original image

    """
    # Remove zeros
    if conf.HEATMAP_THRESHOLD== None:
        lowbound = np.mean(heatmap[heatmap > 0])
        heatmap[heatmap < lowbound] = np.NaN
    else:
         heatmap[heatmap < conf.HEATMAP_THRESHOLD] = np.NaN
    
    # Draw heatmap on top of image
    ax.imshow(heatmap, cmap='inferno', alpha=alpha)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    
    # save the figure 
    if saveFiles:
        fig.savefig(savefilename)
    else:
        fig.show()


def computeHeatmap(conf, gazepoints, fig, ax, dispsize, saveFiles, savefilename=None):
    """
    Return the computed heatmap + draw the heatmap over the images and save the files.
    arguments: 
        - conf : the config object
        - gazepoints: A table containing all the X,Y coordinates of the 
        - fig, ax : figure and its axes with the image already drawn
        - dispsize : The size of the images in pixels
        - saveFiles : if it is asked to save the file or not. 
    optional: 
        - savefilename : The filename to save the image.

    """
    # HEATMAP
    # Gaussian
    if not(conf.IS_GAUSSIAN_KERNEL_LOAD):
        conf.gaussianKernel = gaussian(conf.gaussianKernelWidth, conf.gaussianStd)
        conf.IS_GAUSSIAN_KERNEL_LOAD = conf.gaussianKernel.size !=0;
    
    # matrix of zeroes that will contain the heatmap
    strt = int(conf.gaussianKernelWidth / 2)
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = np.zeros(heatmapsize, dtype=float)
    gwh = conf.gaussianKernelWidth

    ## create heatmapWiener
    for fix in gazepoints:
        # get x and y coordinates of the current fixation
        x = strt + fix[conf.X_COORDINATES_INDEX] - int(gwh / 2) + 30
        y = strt + fix[conf.Y_COORDINATES_INDEX] - int(gwh / 2) - 20
        
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += conf.gaussianKernel[vadj[0]:vadj[1], hadj[0]:hadj[1]] #* gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += conf.gaussianKernel #* gazepoints[i][2]
            
    # Resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    
    #Get a copy of the heatmap 
    heatmapWithZeros = np.copy(heatmap)

    #Display or save the heatmap
    drawHeatmap(heatmap, conf, savefilename, ax, fig, saveFiles=saveFiles)

    #Return the copy of the heatmap
    return heatmapWithZeros

def drawFocusmap(conf, heatmap, fig, ax, savefilename, saveFiles=True):
    """
    Draw the focusmap over the images and eventually save it.
    arguments:
        - ax, fig : should contain a figure with the stimuli already diplayed
        - savefilename : the file which where we need to save the heatmap 
        - conf: the config object
        - heatmap: numpy array containing the heatmap values
    optional:
        - saveFiles: boolean, if we want to save file or just plot them 
    """
    lowbound = 0
    
    if(conf.FOCUSMAP_THRESHOLD == None):
        lowbound = np.mean(heatmap[heatmap > 0])
    else:
        lowbound = threshold
    
    #Display what is above the threshold (by setting to NaN) and set to 0 what's under the threshold
    heatmap[heatmap < lowbound] = 0
    heatmap[heatmap >= lowbound] = np.NaN
    
    #Set the color to gray (to have black value for 0 in the array) and invert axis
    ax.imshow(heatmap, cmap='gray',alpha=1.0)
    ax.invert_yaxis()
    
    #Save the Focusmap if required
    if(saveFiles):
        fig.savefig(savefilename)
    else:
        fig.show()

def generateHeatmapAndFocusmap(fix, imageNo, isParticipant, conf, saveFiles, dispsize, name, participantId = 0, heatmaps = None):
    """
    Method that will generate one heatmap and one focusmap for the 
    arguments:
        - fix : the fixations for 
        - imageNo : The current image number
        - isParticipant : boolean that says if it is a participant or the aggregation of all participant heatmaps
        - conf : the conf object
        - saveFiles: boolean, if we want to save file or just plot them 
        - dispsize: the size of the images in pixels
        - type of heatmap: 0 - normal (text + bubble + other)
                           1 - no text
                           2 - only text
                           3 - only bubble
    optional:
        - participantId: the id of the participant
        - heatmaps: the aggregation of all participant heatmaps
    """
    #Get the basic image (the comics)
    fig1, ax1, fig2, ax2, imgName = getBasicFigures(imageNo, conf)
    
    #Add something in the name of the file if it is a file for one participant only
    suppl=""
    ext =""
    heatmap = None

    if isParticipant:
        #Build the path of the heatmap
        suppl = conf.PARTICIPANT_FOLDER
        ext = "_participant_"+str(participantId)+".png"

        #Generate and get heatmap back
        heatmap = computeHeatmap(conf, fix, fig1, ax1, dispsize, saveFiles, savefilename=conf.HEATMAP_FOLDER+ name+suppl+imgName+ext)

        #Generate Focusmap by thresholding the heatmap
        drawFocusmap(conf, np.copy(heatmap), fig2, ax2, conf.FOCUSMAP_FOLDER+name+suppl+imgName+ext, saveFiles)
    else:
        #Get the aggregated heatmap from all participant and draw them 
        drawHeatmap(np.copy(heatmaps), conf, conf.HEATMAP_FOLDER+name+imgName, ax1, fig1, saveFiles)
        drawFocusmap(conf, heatmaps, fig2, ax2, conf.FOCUSMAP_FOLDER+name+imgName, saveFiles)

    #Close pyplot figures
    pyplot.close(fig1)
    pyplot.close(fig2)
    return heatmap

def generateAllHeatmapsAndFocusmap(conf, saveFiles = True):
    """
    Generate all heatmap and focusmap for every image and for every participant
    arguments:
        - conf ; the cofnig object
    optional: 
        - saveFiles: boolean, if we want to save file or just plot them 
    """
    for key in conf.fixations:
        curImageFixationsNormal = conf.fixations[key]
        curImageFixationsNoText = conf.fixationsNoText[key]
        curImageFixationsOnlyText = conf.fixationsOnlyText[key]
        
        img = ld.getComicImage(conf.imageList, key)
        
        heatmapsNormal = np.zeros((img.height,img.width), dtype=float)
        heatmapsNoText = np.zeros((img.height,img.width), dtype=float)
        heatmapsOnlyText = np.zeros((img.height,img.width), dtype=float)
        
        dispsize = (img.width, img.height)

        for tup in curImageFixationsNormal:
            fix=tup[1]
            heatmapsNormal += generateHeatmapAndFocusmap(fix[:,0:conf.Y_COORDINATES_INDEX+1], key, True, conf, saveFiles, dispsize, "normal/", participantId = tup[0])
            
        for tup in curImageFixationsNoText:
            fix=tup[1]
            heatmapsNoText += generateHeatmapAndFocusmap(fix[:,0:conf.Y_COORDINATES_INDEX+1], key, True, conf, saveFiles, dispsize, "noText/", participantId = tup[0])
        
        for tup in curImageFixationsOnlyText:
            fix=tup[1]
            heatmapsOnlyText += generateHeatmapAndFocusmap(fix[:,0:conf.Y_COORDINATES_INDEX+1], key, True, conf, saveFiles, dispsize, "onlyText/", participantId = tup[0])
        
        generateHeatmapAndFocusmap(None, key, False, conf, saveFiles, dispsize, "normal/", heatmaps=heatmapsNormal)
        generateHeatmapAndFocusmap(None, key, False, conf, saveFiles, dispsize, "noText/", heatmaps=heatmapsNoText)
        generateHeatmapAndFocusmap(None, key, False, conf, saveFiles, dispsize, "onlyText/", heatmaps=heatmapsOnlyText)


def createGazeplot(outputFileByNb, outputFileByDuration, fig, ax, fig2, ax2, gazes, max_radius):
    """
    Create and save gazeplot by fixation duration and by fixation count for one participant
    arguments:
        - outputFileByNb: The path where the gazeplot by fixation count will be saved
        - outputFileByDuration: The path where the gazeplot by fixation duration will be saved
        - fig, ax, fig2, ax2 : Figures and their axes (with the image on it) where the gazeplot will be plot
        - max_radius: the maximal radius of one circle in the gazeplot

    """
    totalDurFix = 0
    totalNbFix = 0
    
    #Compute the gazes total fixation duration and total fixation count
    for g in gazes: 
        totalDurFix += g.fixDuration
        totalNbFix += g.fixNb
        g.computeCenter()
    
    #Plot one circle whose radius depends either on the fixation count (1st gazeplot) or the fixation duration (2nd gazeplot) for each gaze 
    for g in gazes:
        ax.add_artist(pyplot.Circle((g.centerCountX, g.centerCountY), np.log(g.fixNb)/np.log(totalNbFix)*max_radius, color='b', alpha=0.5))
        ax2.add_artist(pyplot.Circle((g.centerDurX, g.centerDurY), np.log(g.fixDuration)/np.log(totalDurFix)*max_radius, color='r', alpha=0.5)) 
    
    #Invert the axis (because fixations coordinates are from upper left corner) and save the figure
    ax.invert_yaxis()
    fig.savefig(outputFileByNb)
    
    ax2.invert_yaxis()
    fig2.savefig(outputFileByDuration)
        
def getGazes(conf):
    """
    Return a dictionnary of the form : imageNo => {Participant ID=> List(Gaze)}
    """
    #Dictionnary that will contain imageID=> {idParticipant => List[Gazes]}
    gazesByImage = {}

    #Iterate over all images
    for k in conf.fixations:
        allFix = conf.fixations[k]
        
        #Iterate over all fixations table of the current image (one per participant)
        for tup in allFix:
            lastAOINb = -2
            curGaze = Gaze(0,0,0)
            curGazeList = []
            
            #Iterate over all fixations of one participant
            for fix in tup[1]:
                #Due to the eye moving fast, it happens that sometimes the fixation duration is 0, we do not take this fixation into account for gazes
                if(fix[conf.FIXDURATION_INDEX]!=0):
                    curAOINb = st.getAOINumberInImage(conf.aoisByImage, k, fix[conf.X_COORDINATES_INDEX], fix[conf.Y_COORDINATES_INDEX])

                    #If we are stil on the same AOI => we are still on the same gaze, we update it and we continue 
                    if(curAOINb==lastAOINb):
                        curGaze.update(fix[conf.X_COORDINATES_INDEX], fix[conf.Y_COORDINATES_INDEX], fix[conf.FIXDURATION_INDEX])
                    else:
                        #If we are in another AOI and it is not the first iteration, we append the gaze to the list  
                        if(lastAOINb != -2):
                            curGazeList.append(curGaze)
                        #And we create another gaze
                        curGaze = Gaze(fix[conf.X_COORDINATES_INDEX], fix[conf.Y_COORDINATES_INDEX], fix[conf.FIXDURATION_INDEX])
                        lastAOINb = curAOINb
                    
            #Append the last gaze to the list        
            curGazeList.append(curGaze)
            
            #Add the list of gazes to the dictionnary (key is the image id and the value is a dictionnary (k=>v) = (participantId =>list of gazes))
            if k in gazesByImage:
                dicOfImage = gazesByImage[k]
                dicOfImage[tup[0]] = curGazeList
                gazesByImage[k] = dicOfImage
            else:
                gazesByImage[k] = {tup[0]: curGazeList}
            
    return gazesByImage

def createAllGazeplots(conf):
    """
    Create all gazeplots 
    """
    gazesByImage = getGazes(conf)

    #Iterate over all images
    for imNo in gazesByImage:
        gazesCurImage = gazesByImage[imNo]
        
        #iterate over all gazes for the current image
        for partNo in gazesCurImage:
            #Get the basic figure with the stimuli on it         
            fig1, ax1, fig2, ax2, imageName = getBasicFigures(imNo, conf)
            
            #Build the save path
            imageNameShort=imageName[0:len(imageName)-conf.EXTENSION_LENGTH]
            outputFileGazeNb = conf.GAZEPLOT_BYFIXNB_FOLDER+imageNameShort+"_participant_"+str(partNo)+".png"
            outputFileGazeDur = conf.GAZEPLOT_BYFIXDUR_FOLDER +imageNameShort+"_participant_"+str(partNo)+".png"

            #Create the gaze plot
            createGazeplot(outputFileGazeNb,outputFileGazeDur, fig1, ax1, fig2, ax2, gazesCurImage[partNo], conf.GAZEPLOT_MAXRADIUS)
            
            #Close the figure
            pyplot.close(fig1)
            pyplot.close(fig2)

def createScanpath(outputFile, fig, ax, fixations, conf):
    """
    Create the scanpath for one participant
    arguments: 
        - outputFile: the outputfile name
        - fig, ax : the figure with the image drawn on it
        - fixations: the fixations for the current participant in the current image
        - conf : the config object
    """

    counter = 1
    #Plot all circle (one for each fixation)
    ax.plot(fixations[:,conf.X_COORDINATES_INDEX], fixations[:,conf.Y_COORDINATES_INDEX], color='red', marker='o',linewidth=5, markersize=20)
    
    #Add text (the counter) to the correct circle (in the order of the fixations)
    for fix in fixations:
        ax.annotate(str(counter), xy=(fix[conf.X_COORDINATES_INDEX], fix[conf.Y_COORDINATES_INDEX]), fontsize=14,fontweight="bold", ha="center", va="center")
        counter+=1
    
    #Invert the correct axis and save the scanpath to the given filename
    ax.invert_yaxis()
    fig.savefig(outputFile)

def createAllScanpaths(conf):
    """
    Create all scanpaths and save them in the configured folder (one scanpath by participant and by images)
    """
    #Iterate over all images
    for imNo in conf.fixations:
        fixationForCurImages = conf.fixations[imNo]

        #Iterate over all fixations in the current image 
        for tup in fixationForCurImages:

            #Get the comic image
            fig1, ax1, fig2, ax2, imageName = getBasicFigures(imNo, conf)
            fix = tup[1]

            #Build the save path
            imageNameShort=imageName[0:len(imageName)-conf.EXTENSION_LENGTH]

            #Create the scanpath and save it.
            createScanpath(conf.SCANPATH_FOLDER+imageNameShort+"_participant_"+str(tup[0])+".png", fig1, ax1, fix[0:,0:2],conf)
            #Close figures
            pyplot.close(fig1)
            pyplot.close(fig2)


def histogramByImage(statCurImage, imageNo, conf, saveFiles, imageName):
    """
    Create histograms for all metrics for one image
    arguments: 
        - statCurImage: The statistics for the current image, dictionnary of the type {aoi => AOIStatistics}
        - imageNo : the current image number
        - conf: the config object
        - saveFiles: need to save the histogram or plot it
        - imageName: the current stimuli name
    """

    #All lists that will contain the metric statistics sorted by ascending AOI id
    listFixNb =[]
    listFixDur =[]
    listGazNumber=[]
    listTtff =[]
    listViewTime =[]
    aoiNbList = []

    #Labels for the axis
    yaxislabels = ["Number", "Milliseconds", "Number", "Milliseconds", "Seconds"]

    #Iterate over the stat by images (in the sorted order of AOI) and add the metric statistics to the correspond list 
    for aoi in sorted(statCurImage.keys()):
        aoiDic = statCurImage[aoi]
        listFixNb.append(aoiDic[conf.categories[0]])
        listFixDur.append(aoiDic[conf.categories[1]])
        listGazNumber.append(aoiDic[conf.categories[2]])
        listTtff.append(aoiDic[conf.categories[3]])
        listViewTime.append(aoiDic[conf.categories[4]])
        aoiNbList.append(aoi)


    allLists = [listFixNb, listFixDur, listGazNumber, listTtff, listViewTime]
    #print("this is allLists:",allLists)
    #print("this is allLists dimensions:", len(allLists))

    #Create the number of subplots corresponding to the number of categories
    fig, axs = pyplot.subplots(len(conf.categories) + 1,figsize=(15,30), sharey=False)

    #Iterate over all metrics and create one subplots by metric
    for nb in range(len(conf.categories)):
        curList = allLists[nb]
        meanList = []
        errList = []

        #Get over each AOI and get mean and std dev for the current metric 
        for aoiNb in aoiNbList:
            curItem = curList[aoiNb]
            
            if any([str(val) == "nan" for val in curItem.values()]):
                curItem = {'mean': 0.0, 'stdDev': 0.0, 'boundUp': 0, 'boundDown': 0}
                
            meanList.append(curItem["mean"])
            errList.append(curItem["stdDev"])

        
        #Setup the current subplot
        """
        if(len(meanList)!=0 or len(errList)!=0):
            
            maxs = np.max(np.array(meanList)+np.array(errList))
            if(np.isnan(maxs).any()):
                maxNb=100 
            else:
                maxNb = math.ceil(maxs)
            ysteps = max(int(maxNb/7),1)
        """
        
        try:
            maxs = np.max(np.array(meanList)+np.array(errList))
            maxNb = math.ceil(maxs)
            ysteps = max(int(maxNb/7),1)
        
            axs[nb].set(xlabel="AOI ID in image",ylabel = yaxislabels[nb])
            axs[nb].set_xticks(aoiNbList)
            axs[nb].set_yticks(range(0,maxNb+ysteps,ysteps))
            axs[nb].set_xticklabels((aoiNbList))
            axs[nb].set_ylim(bottom=0.)
            axs[nb].set_title(conf.categories[nb] + " for image n°" + str(imageNo) + " ("+imageName+")")
            axs[nb].bar(aoiNbList, meanList, yerr=errList,alpha=0.5, error_kw=dict(ecolor='grey', lw=2, capsize=5))
            
        except ValueError:  #raised if `y` is empty.
            pass

        for i, v in enumerate(meanList):
                axs[nb].text(i-0.1, v , str(v), color='black', va="bottom", ha="center",rotation=90, fontsize=10, fontweight='bold')

    
    #Get the fastest reached AOIs
    fastest_reached_aois = shortest_time_to_first_fixation(statCurImage, conf, imageNo)

    #Create a new list to store the AOI names, their respective supercategory, and their time to first fixation
    fastest_reached_aois_with_supercat = [(aoi , conf.aoisByImage[imageNo][aoi].supercategory) for aoi in fastest_reached_aois]
    print("this is fastest_reached_aois_with_supercat:",fastest_reached_aois_with_supercat)
    # Create a list of colors for each supercategory
    supercat_colors = [conf.supercategoriesColors[conf.supercategories.index(supercat)] for _, supercat in fastest_reached_aois_with_supercat]

    #Create the histogram of fastest reached AOIs
    axs[len(conf.categories)].set(xlabel="AOI ID and Supercategory", ylabel="Milliseconds")
    axs[len(conf.categories)].set_title("Fastest Reached AOIs for image n°" + str(imageNo) + " ("+imageName+")")
    axs[len(conf.categories)].set_xticks(range(0, len(fastest_reached_aois_with_supercat)))
    axs[len(conf.categories)].set_xticklabels([(str(aoi[0]) + " - " + aoi[1]) for aoi in fastest_reached_aois_with_supercat], rotation=45, ha='right')
    
    # Set the color of each bar to the corresponding supercategory color
    axs[len(conf.categories)].bar(range(0, len(fastest_reached_aois_with_supercat)), [statCurImage[aoi]['Time to first fixation']['mean'] for aoi, _ in fastest_reached_aois_with_supercat], alpha=0.5, color=supercat_colors)
    
    #Make the font biiger Adjust the layout and display the plot
    pyplot.rcParams.update({'font.size': 25})
    pyplot.subplots_adjust(hspace=0.5)
    fig.tight_layout()


    #Save the histograms or plot them
    if(saveFiles):
        fig.savefig(conf.HISTOGRAMS_FOLDER+"image/"+imageName+"_histograms.png")
    else:
        fig.show()

    #Close the figures
    pyplot.close(fig)
    
    
def shortest_time_to_first_fixation(statCurImage, conf, imageNo):
    """
    Helper function for the histogramsByImage function.
    Returns a list of the id's of the ten AOIs that have the shortest time to first fixation
    amongst the AOIs having a time to first fixation different than zero.
    arguments: 
        - statCurImage: The statistics for the current image, dictionnary of the type {aoi => AOIStatistics}
        - conf: The configuration object
        - imageNo: The current image number
    """
    
    non_zero_tff_aois = []   
    for aoi in statCurImage.keys():
        # Skip AOIs with a time to first fixation of zero or with undesired supercategories
        aoi_supercategory = conf.aoisByImage[imageNo][aoi].supercategory
        if any([str(val) == "nan" for val in statCurImage[aoi]["Time to first fixation"].values()]) or aoi_supercategory in ['OTHER', 'BACKGROUND']:
            continue
        else: 
            non_zero_tff_aois.append(aoi)
    
    # Sort the AOIs by ascending time to first fixation
    sorted_aois_by_tff = sorted(non_zero_tff_aois, key=lambda aoi: statCurImage[aoi]["Time to first fixation"]['mean'])
    
    # Return the top 10 AOIs with the shortest time to first fixation
    return sorted_aois_by_tff[:10]

    
    
def histogramBySuperCategory(statCurImage, imageNo, conf, saveFiles, imageName):
    """
    Create histograms for all metrics for one image grouped by supercategory
    arguments: 
        - statCurImage: The statistics for the current image, dictionary of the type {aoi => AOIStatistics}
        - imageNo : the current image number
        - conf: the config object
        - saveFiles: need to save the histogram or plot it
        - imageName: the current stimuli name
    """

    # Initialize dictionaries of the form {supercategory => {category => []}}
    supercategoryMetrics = {}
    for supercategory in conf.supercategories:
        if(supercategory == "OTHER"):
            continue
        supercategoryMetrics[supercategory] = {category: [] for category in conf.categories}

    # Iterate over the statistics by AOI and add the metric statistics to the corresponding supercategory
    for aoi in statCurImage:
        aoiStats = statCurImage[aoi]
        supercategory = conf.aoisByImage[imageNo][aoi].supercategory # get the supercategory of the AOI
        for category in conf.categories:
            supercategoryMetrics[supercategory][category].append(aoiStats[category])

    # Create lists of metrics sorted by supercategory
    allLists = []
    

    for category in conf.categories:
        categoryList = []
        for supercategory in conf.supercategories:
            if(supercategory == "OTHER"):
                continue
            supercategoryList = supercategoryMetrics[supercategory][category]
            # sum up the fixation values for each AOI in the current supercategory and append to the categoryList
            categoryList.append(sum([sum(aoi.values()) for aoi in supercategoryList]))
        allLists.append(categoryList)

    # Labels for the axis
    yaxislabels = ["Number", "Milliseconds", "Number", "Milliseconds", "Seconds"]

    # Create the number of subplots corresponding to the number of categories
    fig, axs = pyplot.subplots(len(conf.categories), figsize=(20, 30), sharey=False)

    count = 0
    # Iterate over all metrics and create one subplot by metric
    for nb in range(len(conf.categories)):
        #skip the time to first fixation metric
        if(nb == 3):
            continue

        curList = allLists[nb]
        
        newNb = nb
        
        if (nb > 3):
            newNb = nb - 1
            
        # Setup the current subplot
        try:
            maxs = np.max(np.array(curList))
            maxNb = math.ceil(maxs)
            ysteps = max(int(maxNb/7), 1)

            axs[newNb].set(xlabel="Super Category ID", ylabel=yaxislabels[nb])
            axs[newNb].set_xticks(range(len(conf.supercategories)))
            axs[newNb].set_xticklabels(conf.supercategories)
            axs[newNb].set_yticks(range(0, maxNb + ysteps, ysteps))
            axs[newNb].set_ylim(bottom=0.)
            axs[newNb].set_title(conf.categories[nb] + " for image n°" + str(imageNo) + " (" + imageName + ")")
            axs[newNb].bar(range(len(conf.supercategories)),  curList, alpha=0.5, error_kw=dict(ecolor='grey', lw=2, capsize=5), color=[conf.supercategoriesColors[conf.supercategories.index(cat)] for cat in conf.supercategories])
            
        except ValueError:  # raised if y is empty.
            pass

        for i, v in enumerate(curList):
                axs[newNb].text(i-0.1, v , str(v), color='black', va="bottom", ha="center",rotation=90, fontsize=10, fontweight='bold')
                
    # add the area percentage subplot to the figure
    area_percentage = calculate_area_percentage_by_supercategory(conf.aoisByImage, imageNo)
    supercategory_names = list(area_percentage.keys())
    area_percentages = list(area_percentage.values())

    # Index of the new subplot
    new_subplot_index = len(conf.categories) - 1

    # Create the area percentage histogram as a subplot
    axs[new_subplot_index].set(xlabel="Super Category ID", ylabel="Percentage of Area")
    axs[new_subplot_index].set_xticks(range(len(supercategory_names)))
    axs[new_subplot_index].set_xticklabels(supercategory_names)
    axs[new_subplot_index].set_ylim(bottom=0., top=100.)
    axs[new_subplot_index].set_title("Percentage of Area by Super Category for image n°" + str(imageNo) + " (" + imageName + ")")
    axs[new_subplot_index].bar(range(len(supercategory_names)), area_percentages, alpha=0.5, color=[conf.supercategoriesColors[conf.supercategories.index(cat)] for cat in supercategory_names])

    for i, v in enumerate(area_percentages):
        axs[new_subplot_index].text(i - 0.1, v, f"{v:.1f}%", color='black', va="bottom", ha="center", rotation=90, fontsize=10, fontweight='bold')


    #Make the font biiger Adjust the layout and display the plot
    pyplot.rcParams.update({'font.size': 25})
    pyplot.subplots_adjust(hspace=0.5)
    fig.tight_layout()

    #Save the histograms or plot them
    if(saveFiles):
        fig.savefig(conf.HISTOGRAMS_FOLDER+"/category/"+imageName+"_histograms.png")
    else:
        fig.show()
    
    #Close the figures
    pyplot.close(fig)

def calculate_area_percentage_by_supercategory(aois_by_image, image_no):
    """
    Helper function for HistogramsBySuperCategory.
    Calculate the percentage of area taken by each supercategory in the image.

    Arguments:
        - aois_by_image: A dictionary that maps image numbers to AOIs (the same format as conf.aoisByImage)
        - image_no: The current image number

    Returns:
        A dictionary that maps supercategory names to their area percentages in the image.
    """
    # Initialize a dictionary to store the total area for each supercategory
    supercategory_areas = {}

    # Iterate over AOIs in the current image
    for aoi in aois_by_image[image_no].values():
        supercategory = aoi.supercategory

        # Calculate the area of the current AOI
        aoi_area = aoi.getPixelNumber()

        # Add the AOI area to the corresponding supercategory
        if supercategory in supercategory_areas:
            supercategory_areas[supercategory] += aoi_area
        else:
            supercategory_areas[supercategory] = aoi_area

    # Calculate the total area of all AOIs in the image
    total_area = sum(supercategory_areas.values())

    # Calculate the area percentage for each supercategory
    supercategory_percentage = {supercategory: (area / total_area) * 100 for supercategory, area in supercategory_areas.items()}

    return supercategory_percentage

    
def createAllHistograms(statistics, conf, saveFiles = True):
    """
    Generate histogram for every images
    arguments:
        - statistics: the global statistics by images. Dictionnary of the form {imageNo => { 
                                                                                        aoi => AOIStatistics
                                                                                        }
                                                                                }
        - conf : the config object
    optional:
        - saveFiles: save or plot the files
    """
    #Iterate over all images and create one histogram by image.
    for imageNo in sorted(statistics.keys()):
        histogramByImage(statistics[imageNo], imageNo, conf,  saveFiles=saveFiles, imageName = ld.getComicImage(conf.imageList,imageNo).imageNameShort)
        histogramBySuperCategory(statistics[imageNo], imageNo, conf,  saveFiles=saveFiles, imageName = ld.getComicImage(conf.imageList,imageNo).imageNameShort)


def saveMap(map, imNo, partNo, conf, binMap=False, salMap=False, combBinMap=False):
    """
    Method that will save a saliency, binary or combined binary map to disk (in the configured files).
    arguments:
        - map : the map that need to be saved (numpy array)
        - imNo : the current image number
        - partNo : the current participant number
        - conf : the config object
    optional: 
        - binMap : if the map is a binary map
        - salMap : if the map is a saliency map
        - combBinMap : if the map is a combined binary map
    """
    #Generate the basic figure with the good axis
    fig1, ax1, fig2, ax2, imageName = getBasicFigures(imNo, conf, drawImage=False)
    ax1.imshow(map, cmap='gray',alpha=1.0)
    ax1.invert_yaxis()
    fig1.show()

    #If it is not one of the 3 types of map, we return without doing anything
    if not(salMap) and not(binMap) and not(combBinMap):
        return

    #Build the correct path 
    path = ""
    part = imageName+"_participant_"+str(partNo) +".png"
    if binMap:
        path = conf.BINARY_MAP_FOLDER

    if salMap:
        path = conf.SALIENCY_MAP_FOLDER

    if combBinMap:
        path = conf.COMBINED_BINARY_MAP_FOLDER

    #Save the map to the disk
    fig1.savefig(path+part)

    #Close figures
    pyplot.close(fig1)
    pyplot.close(fig2)


def blur(binMap,sigma=33.33, kernelSize = (201,201)):
    """
    Return the blurred binary map using a gaussian blur.

    arguments: 
        - binMap : the binary map to be blurred
    optional:
        - sigma: standard deviation into the gaussian kernel
        - kernelSize : size of the Gaussian kernel
    """
    return cv2.GaussianBlur(binMap, kernelSize, sigma)

def computeBinaryMap(fixation, imageWidth, imageHeight, conf):
    """
    Compute and return the binary map derived from the fixations 
    arguments
        - fixation : a numpy array containing all fixations for the current participant
        - imageWidth : image width in pixel
        - imageHeight : image height in pixel
        - conf : the config object. 
    """
    fixation = fixation.astype(int)
    binArr = np.zeros((imageHeight,imageWidth))

    for fix in fixation:
        if fix[conf.Y_COORDINATES_INDEX]<imageHeight and fix[conf.X_COORDINATES_INDEX]<imageWidth:
            binArr[fix[conf.Y_COORDINATES_INDEX]][fix[conf.X_COORDINATES_INDEX]] = 1.0
    return binArr

"""
Compute the saliency map from one binary map and saves it if necessary.
"""
def computeSaliencyMap(binMap, imNo=0, saveFiles = False, partNo=0, conf=None):
    """
    Return the saliency map

    arguments
        - binMap : The binary map    from which we want to compute the saliency map
    optional 
        - imNo : current image number (needed when we want to save files)
        - saveFiles : if the saliency map need to be saved
        - partNo : current participant number (if we want to save files)
        - conf: conf object (if we want to save files)
    """

    salMap = blur(binMap)*255

    if(saveFiles):
        saveMap(salMap, imNo, partNo, conf, salMap =True)
    return salMap


def computeAllBinaryMap(conf, saveFiles=False):
    """
    This method computes the saliency map for each image of all participants and returns a dictionnary such that (key => v) = (imageNo => List((participantId, saliencyMap)))
    arguments:
        - conf : config object
    optionals: 
        - saveFiles : if we want to save the file to the configured folder.
    """
    binMapDictionnary = {}

    for imageNo in conf.fixations:
        curImg = conf.imageList[imageNo]
        #List of tuple (participantNo, saliencyMap)
        listTup = []
        for tup in conf.fixations[imageNo]:
            binMap = computeBinaryMap(tup[1], curImg.width, curImg.height, conf)
            
            listTup.append((tup[0],binMap))
            
            if(saveFiles):
                saveMap(binMap, imageNo, tup[0], conf, binMap=True)

        binMapDictionnary[imageNo] = listTup
    return binMapDictionnary

"""
This method takes as argument the false and true positive rate lists and plot the ROC Curve.
"""
def plotROCCurve(fp,tp):
    pyplot.plot(fp,tp)
    pyplot.show()


