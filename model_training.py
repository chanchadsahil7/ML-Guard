
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

"""
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors':[1,3,5,7], 'weights':['uniform',"distance"],"algorithm":["auto", "ball_tree", "kd_tree", "brute"]}
knn = KNeighborsClassifier()
model = GridSearchCV(knn, parameters)
"""


def loadTrainDataAndTrainModel():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    model = KNeighborsClassifier(n_neighbors=5,weights="distance",algorithm="auto")      ## best parameter values

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:                                                                                 # if file could not be opened
        print("Error, unable to open classifications.txt, exiting program\n")                # show error message
        os.system("pause")
        return False                                                                        # and return False
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:                                                                                 # if file could not be opened
        print("Error, unable to open flattened_images.txt, exiting program\n")               # show error message
        os.system("pause")
        return False                                                                        # and return False
    # end try

    model.fit(npaFlattenedImages, npaClassifications)

    #print(model.best_params_)                                                         ## printing the best parameters

    # save the model to disk
    filename = 'finalized_model.sav'
    joblib.dump(model, filename,compress=3)
    return True

if __name__ == "__main__":
    modelTrainingSuccessful = loadTrainDataAndTrainModel()  # attempt KNN training
    if modelTrainingSuccessful == False:  # if KNN training was not successful
        print("\nError: model traning was not successful\n")  # show error message
    else:
        print("\nModel traning successful\n")  # show success message
    #endif
