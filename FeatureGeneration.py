import os
import time
import scipy.io as sio
import pandas as pd
import numpy as np
import scipy.signal as signal


rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"

sourceType = "Train"
sourcePath = rootDir + "Data1/train_1/"
destFile = rootDir + "train_1.csv"

sourceType = "Test"
sourcePath = rootDir + "Data1/test_1/"
destFile = rootDir + "test_1.csv"

def create_features_file(sourcePath, sourceType, destFile):
    
    startTime = time.time()

    samplingRate = 400
    Nyquist = 0.5 * samplingRate
    timeWindows = [0, 60000, 120000, 180000, 240000]
    freqBands = [0, 4, 8, 16, 32, 64, Nyquist]
    
    numFeatures = 16 * (len(timeWindows)-1) * (len(freqBands)-1)
    
    colNames = ["File"]
    if sourceType == "Train":
        colNames.append("Class")
        
    for i in range(1,numFeatures+1):
        colNames.append("Feature" + str(i))

    features = pd.DataFrame(columns = colNames)

    fileNames = os.listdir(sourcePath)
    i = 1
    for fileName in fileNames:
        if (i%50 == 0): print("Processing file {} of {}".format(i, len(fileNames)))
        try:
            fileContents = sio.loadmat(sourcePath + fileName, struct_as_record=False)
            fileContents = fileContents["dataStruct"][0,0]
            eegData = fileContents.data

            features.loc[i, "File"] = fileName
            if sourceType == "Train":
                features.loc[i, "Class"] = fileName.split(".")[0][-1]
            
            features.loc[i, "Feature1":colNames[-1]] = generate_features(eegData, samplingRate, timeWindows, freqBands)
            
            i+=1;
            #if i==100: break

        except ValueError:
            print("Could not process file: " + fileName)

    features.to_csv(destFile)

    print("Processing time: {:.2f} seconds".format(time.time() - startTime))

    return


def generate_features(eegData, samplingRate, timeWindows, freqBands):

    from IPython.core.debugger import Tracer; dbg_breakpoint = Tracer()
    
    numChannels = eegData.shape[1]
    numWindows = len(timeWindows)-1
    numBands = len(freqBands)-1
    
    features = np.zeros(numChannels * numWindows * numBands)
    
    for channel in range(numChannels):
        for i in range(numWindows):
            channelData = eegData[timeWindows[i]:timeWindows[i+1],channel]
            freq, PSD = signal.periodogram(channelData, samplingRate)
            totalPSD = sum(PSD)
            for j in range(numBands):
                #dbg_breakpoint()
                freqFilter = np.logical_and(freq >= freqBands[j], freq < freqBands[j+1])
                featNumber = channel*numWindows*numBands + i*numBands + j
                features[featNumber] = sum(PSD[freqFilter])/totalPSD
                    
    return features


            







