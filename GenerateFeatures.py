import os
import scipy.io as sio
import pandas as pd
import numpy as np

sourcePath = "C:/Users/Rodrigo/Documents/Data Science/Seizure Prediction/Data/Data1/train_1/"
destFile = "C:/Users/Rodrigo/Documents/Data Science/Seizure Prediction/Data/train1.csv"

colNames = ["File", "Class"]
for i in range(1,33):
    colNames.append("Feature" + str(i))

features = pd.DataFrame(columns = colNames)

#fileNames = os.listdir(filePath)
#for i in range(1,3):
#    fileName = fileNames[i]
#    print(fileName)

i = 1
for fileName in os.listdir(sourcePath):
    try:
        fileContents = sio.loadmat(sourcePath + fileName, struct_as_record=False)
        fileContents = fileContents["dataStruct"]
        fileContents = fileContents[0,0]
        eegData = pd.DataFrame(fileContents.data)

        features.loc[i, "File"] = fileName
        features.loc[i, "Class"] = fileName.split(".")[0][-1]
        features.loc[i, "Feature1":"Feature16"] = eegData.mean().values
        features.loc[i, "Feature17":"Feature32"] = eegData.std().values
        i+=1
        
    except ValueError:
        print("Could not process file: " + fileName)
        

features.to_csv(destFile)
    
