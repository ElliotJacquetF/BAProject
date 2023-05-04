import os
class Config: 
    #Current version number of the software
    CURRENT_VERSION = "1.2"
    print("Current version " +CURRENT_VERSION) 

    #Use IDT by default to parse gazepoints into fixations
    useIDT = True

    ## Fixation array
    X_COORDINATES_INDEX = 0
    Y_COORDINATES_INDEX = 1
    TIMESTAMP_INDEX = 2
    FIXDURATION_INDEX = 3
    TIMEFROMSTART_INDEX = 4

    #List of categories of interest
    categories = ["Fixation count", "Fixation duration", "Gaze count", "Time to first fixation", "Viewing Time"]

    #List of supercategories
    supercategories = ["OBJECT", "ANIMAL", "BACKGROUND", "CHARACTER", "TEXT"]
    #List of colors for each supercategory
    supercategoriesColors = ["#FFFF00", "#FFA500", "#808080", "#FF0000", "#0000FF"]

    #Fixations files constants
    EXTENSION_LENGTH = 4
    NB_LINES_BETWEEN_AOI_LIST = 15
    START_INDEX_AOI_LIST = 4

    ## FOLDER
    RESOURCES = "resources/"

    #Needed folder
    STIMULI = RESOURCES + "stimuli/"
    GAZEPOINTS = RESOURCES + "gazepoints/"
    OUTPUTS = RESOURCES + "outputFiles/"

    #Visualisations output folder
    HEATMAP_FOLDER = OUTPUTS + "heatmaps/"
    GAZEPLOT_FOLDER = OUTPUTS + "gazeplots/"
    GAZEPLOT_BYFIXNB_FOLDER = GAZEPLOT_FOLDER + "byFixCount/"
    GAZEPLOT_BYFIXDUR_FOLDER = GAZEPLOT_FOLDER + "byFixDuration/"
    SCANPATH_FOLDER = OUTPUTS + "scanpaths/"
    FOCUSMAP_FOLDER = OUTPUTS + "focusmaps/"
    HISTOGRAMS_FOLDER = OUTPUTS + "histograms/"
    BINARY_MAP_FOLDER = OUTPUTS + "binaryMap/"
    COMBINED_BINARY_MAP_FOLDER = OUTPUTS + "binaryMapNObservers/"
    SALIENCY_MAP_FOLDER = OUTPUTS + "saliencyMap/"
    DEEP_GAZE = OUTPUTS + "deep_gaze/"
    ICF = OUTPUTS + "icf/"
    PARTICIPANT_FOLDER = "participants/"
    AOI_FOLDER = OUTPUTS + "aoisByImage/"
    AUC_FOLDER = OUTPUTS + "auc/"

    #Lists of all folders needed to run the program properly
    FOLDER_LISTS = [RESOURCES, GAZEPOINTS, STIMULI, OUTPUTS, DEEP_GAZE, ICF, HEATMAP_FOLDER, HEATMAP_FOLDER+PARTICIPANT_FOLDER, GAZEPLOT_FOLDER, GAZEPLOT_BYFIXDUR_FOLDER, GAZEPLOT_BYFIXNB_FOLDER, SCANPATH_FOLDER, FOCUSMAP_FOLDER, FOCUSMAP_FOLDER+PARTICIPANT_FOLDER, HISTOGRAMS_FOLDER, BINARY_MAP_FOLDER, COMBINED_BINARY_MAP_FOLDER, SALIENCY_MAP_FOLDER, AOI_FOLDER, AUC_FOLDER]

    #AOIS Loading
    ARE_AOIS_LOAD = False
    allAOIS= []
    aoisByImage = {}

    #Images Loading 
    ARE_IMAGES_LOAD = False
    imageList = []

    #Fixations loading
    ARE_FIXATIONS_LOAD = False
    fixations = {}

    #Gaussian kernel parameters for heatmaps
    IS_GAUSSIAN_KERNEL_LOAD = False
    gaussianKernelWidth = 200
    gaussianStd = gaussianKernelWidth/6
    gaussianKernel = None

    #AOI file
    AOIS_FILE = RESOURCES + "instances_default_modified_2.json"

    ## Thresholds
    # IDT Thresholds : These thresholds come from the Paper : One algorithm to rule them all ? An evaluation and discussion of ten eye movement event-detection algortihms, published in Behavior Research Methods
    dispersion_thresold_degree = 1.5  # 
    duration_threshold = 100  # In millisecond
    distance_to_screen = 65 #cm
    heightOnScreen = 29.7 #cm

    #Visualisations Threshold
    HEATMAP_THRESHOLD = None
    FOCUSMAP_THRESHOLD = None
    GAZEPLOT_MAXRADIUS = 80

    #List of saliency prediction models we use 
    MODEL_LIST = ["DeepGazeII.ckpt", "ICF.ckpt"]
    MODEL_EXTENSIONS_LENGTH = 5
    DEEP_GAZE_RESOURCES_FOLDER = RESOURCES + "deepGazeFiles/"

    #This method create the folder tree hierarchy needed to use the application
    ## If the folder already exists, it won't be removed 
    def __init__(self):
        for fold in self.FOLDER_LISTS:
            try:
                os.makedirs(fold)
            except:
                pass

