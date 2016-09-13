# -------------------------------------------------------------------------------------- #
# Create all training and test files
# -------------------------------------------------------------------------------------- #
def create_all_files():
    import time
    import pandas as pd

    startTime = time.time()

    dataSet1 = {"Type": "Train", "Source": "Data1/train_1/", "Output Name": "train_1.csv"}
    dataSet2 = {"Type": "Train", "Source": "Data2/train_2/", "Output Name": "train_2.csv"}
    dataSet3 = {"Type": "Train", "Source": "Data3/train_3/", "Output Name": "train_3.csv"}
    dataSet4 = {"Type": "Test" , "Source": "Data1/test_1/" , "Output Name": "test_1.csv"}
    dataSet5 = {"Type": "Test" , "Source": "Data2/test_2/" , "Output Name": "test_2.csv"}
    dataSet6 = {"Type": "Test" , "Source": "Data3/test_3/" , "Output Name": "test_3.csv"}
    
    dataSets = [dataSet1, dataSet2, dataSet3, dataSet4, dataSet5, dataSet6]
    rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"

    for dataSet in dataSets:
        sourceType = dataSet["Type"]
        sourcePath = rootDir + dataSet["Source"]
        destFile = rootDir + dataSet["Output Name"]
        print("Creating file {}".format(dataSet["Output Name"]))
        features = create_features_dataframe(sourcePath, sourceType)
        features.to_csv(destFile, index = False)

    print("Creating file {}".format("train_all.csv"))
    train_1 = pd.read_csv(rootDir + "train_1.csv")           
    train_2 = pd.read_csv(rootDir + "train_2.csv")           
    train_3 = pd.read_csv(rootDir + "train_3.csv")           
    train_all = pd.concat([train_1, train_2, train_3])
    train_all.to_csv(rootDir + "train_all.csv", index = False)

    print("Creating file {}".format("test_all.csv"))
    test_1 = pd.read_csv(rootDir + "test_1.csv")           
    test_2 = pd.read_csv(rootDir + "test_2.csv")           
    test_3 = pd.read_csv(rootDir + "test_3.csv")           
    test_all = pd.concat([test_1, test_2, test_3])
    test_all.to_csv(rootDir + "test_all.csv", index = False)

    print("Total processing time: {:.2f} seconds".format(time.time() - startTime))

    return
        
        
        
# -------------------------------------------------------------------------------------- #
# Create features dataframe
# -------------------------------------------------------------------------------------- #
def create_features_dataframe(sourcePath, sourceType):
    import os
    import scipy.io as sio
    import pandas as pd
    
    samplingRate = 400
    Nyquist = 0.5 * samplingRate
    timeWindows = [0, 120000, 240000]
    freqBands = [0, 4, 8, 32, Nyquist]
    
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
        if (i%100 == 0): print("    Processing file {} of {}".format(i, len(fileNames)))
        try:
            fileContents = sio.loadmat(sourcePath + fileName, struct_as_record=False)
            fileContents = fileContents["dataStruct"][0,0]
            eegData = fileContents.data

            features.loc[i, "File"] = fileName
            if sourceType == "Train":
                features.loc[i, "Class"] = fileName.split(".")[0][-1]
            
            features.loc[i, "Feature1":colNames[-1]] = generate_features(eegData, samplingRate, timeWindows, freqBands)
            
            i+=1;
            #if i==10: break

        except ValueError:
            print("    Could not process file: " + fileName)

    # Fill missing values with median of respective feature
    for i in range(1, numFeatures + 1):
        colName = "Feature" + str(i)
        features.loc[features[colName].isnull(), colName] = features[colName].dropna().median()

    return features


# -------------------------------------------------------------------------------------- #
# Generate features 
# -------------------------------------------------------------------------------------- #
def generate_features(eegData, samplingRate, timeWindows, freqBands):

    import scipy.signal as signal
    import numpy as np
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


            







