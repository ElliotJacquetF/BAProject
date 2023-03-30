import Metrics as me
from matplotlib import image, pyplot, lines
import numpy as np
import Visualisations as vs

from scipy.ndimage import zoom
from scipy.special import logsumexp
import tensorflow.compat.v1 as tf

"""
This method uses the DeepGaze and ICF models from https://deepgaze.bethgelab.org to predict the saliency maps for each stimuli
"""
def predictAllSaliencyMaps(conf, saveFiles=True):
    """
    Return a dictionnary of dictionnary: i.e. a dictionnary in the following format {modelName => {imageNo => saliencyMap}}
    """
    predSalMap = {}
    tf.disable_eager_execution()

    for check_point in conf.MODEL_LIST:
        #Load the model from the checkpoints downloaded on the website.
        tf.reset_default_graph()
        check_point_path = conf.DEEP_GAZE_RESOURCES_FOLDER + check_point
        new_saver = tf.train.import_meta_graph( '{}.meta'.format(check_point_path))

        curSaliencyMaps = {}

        with tf.Session() as sess:
            #Restore the weights
            new_saver.restore(sess, check_point_path)

            #For each image, create the saliency map
            for imObj in conf.imageList:
                #Load image, and makes its values in the range [0,255]
                img = image.imread(conf.STIMULI + imObj.imageName)
                img2 = img.astype(np.int32)*255

                # BHWC, three channels (RGB). So we need to remove the alpha channel from the images we laod from the disk
                image_data = img2[np.newaxis, :, :, 0:3] 

                #Predict the log_density
                input_tensor = tf.get_collection('input_tensor')[0]
                log_density_wo_centerbias = tf.get_collection('log_density_wo_centerbias')[0]
                    
                log_density_prediction = sess.run(log_density_wo_centerbias, {
                    input_tensor: image_data
                })

                #Get back the distribution
                pred = np.exp(log_density_prediction[0,:,:,0])

                #Save the image if required
                if saveFiles:
                    fig1, ax1, fig2, ax2, imageName = vs.getBasicFigures(imObj.imageNo, conf, drawImage=False, alpha=0.2)
                    ax1.invert_yaxis()
                    ax1.matshow(pred, cmap='gist_gray',alpha=1)
                    
                    path = conf.DEEP_GAZE if  check_point==conf.MODEL_LIST[0] else conf.ICF
                    
                    fig1.savefig(path+imObj.imageName)
                    pyplot.close(fig1)
                    pyplot.close(fig2)

                curSaliencyMaps[imObj.imageNo] = pred
        predSalMap[check_point] = curSaliencyMaps
    return predSalMap


def computeModelScores(predictions, sumBinMapsByImages, conf):
    """
    Compute AUC, KL, NSS scores for each model and for each image. and return a dictionnary of the form {imageNo => {
                                                                                                            modelType => {
                                                                                                                metricType => Score
                                                                                                            }
                                                                                                        }
                                                                                                    } 
    arguments:
        - predictions : the predicted saliency maps dictionnary of the form : {modelType => {imageNo => Predicted saliency map}}
        - sumBinMapsByImages : the sum of all binary maps over all participants for each image (Dictionnary: {imageNo => sumBinMap})
        - conf : the config object 
    """
    allModelScores = {}
    meanScores = {}
    for check_point in conf.MODEL_LIST:
        curModelScores = {}
        predictionByModel = predictions[check_point]
        NSSScores = []
        AUCScores = []
        KLScores = []

        for imageNo in predictionByModel:
            currentImageScore = {}
            gt = np.zeros(sumBinMapsByImages[imageNo].shape)
            gt[sumBinMapsByImages[imageNo] >=1] = 1

            auc = me.auc_judd(predictionByModel[imageNo], gt)
            kl = me.KLdiv(predictionByModel[imageNo], gt)
            nss = me.NSS(predictionByModel[imageNo], gt)

            AUCScores.append(auc)
            KLScores.append(kl)
            NSSScores.append(nss)

            currentImageScore["AUC"] = auc
            currentImageScore["KL"] = kl
            currentImageScore["NSS"] = nss

            curModelScores[imageNo] = currentImageScore

        aucArray = np.array(AUCScores)
        klArray = np.array(KLScores)
        nssArray = np.array(NSSScores)

        meanScores[check_point] = {"AUC":(np.mean(aucArray),np.std(aucArray)),"KL":(np.mean(klArray),np.std(klArray)),"NSS":(np.mean(nssArray),np.std(nssArray))}
        allModelScores[check_point] = curModelScores
    return allModelScores, meanScores   


def printModelScores(modelScores, meanScores, conf):
    """
    Print the model scores
    arguments:
        - modelScores : Dictionnary of the form {imageNo => {
                                                        modelType => {
                                                            metricType => Score
                                                        }
                                                    }
                                                } 
        - conf : the config object
    """
    for model in conf.MODEL_LIST:
        print("===========================================================================")
        print("Average scores for model "+ model)
        scores = meanScores[model]
        for s in scores:
            print("\t"+s+" : "+str(round((scores[s])[0],3))+" (STD: "+str(round((scores[s])[1],3))+")")
    for img in conf.imageList:
        print("===========================================================================")
        print("Model scores for image n°"+str(img.imageNo)+" ("+img.imageNameShort+")")
        for model in conf.MODEL_LIST:
            print("\t"+model +" gets :")
            scores = (modelScores[model])[img.imageNo]
            for k in scores:
                print("\t\t"+k+" score : "+str(round(scores[k],3)))
            print("")
        print("===========================================================================")
        print("")
