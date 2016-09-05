import os
import scipy.io as sio
import pandas as pd
import numpy as np

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"

sourceType = "Train"
sourcePath = rootDir + "Data1/train_1/"
destFile = rootDir + "train_1.csv"

sourceType = "Test"
sourcePath = rootDir + "Data1/test_1/"
destFile = rootDir + "test_1.csv"

def GenerateFeatures(sourcePath, sourceType, destFile):

    colNames = ["File"]
    if sourceType == "Train":
        colNames.append("Class")
        
    for i in range(1,33):
        colNames.append("Feature" + str(i))

    features = pd.DataFrame(columns = colNames)

    i = 1
    for fileName in os.listdir(sourcePath):
        try:
            fileContents = sio.loadmat(sourcePath + fileName, struct_as_record=False)
            fileContents = fileContents["dataStruct"]
            fileContents = fileContents[0,0]
            eegData = pd.DataFrame(fileContents.data)

            features.loc[i, "File"] = fileName
            if sourceType == "Train":
                features.loc[i, "Class"] = fileName.split(".")[0][-1]
            features.loc[i, "Feature1":"Feature16"] = eegData.mean().values
            features.loc[i, "Feature17":"Feature32"] = eegData.std().values
            i+=1

        except ValueError:
            print("Could not process file: " + fileName)

    features.to_csv(destFile)

    return


    
