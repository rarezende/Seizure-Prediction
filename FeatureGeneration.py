# -------------------------------------------------------------------------------------- #
# Create all training and test files
# -------------------------------------------------------------------------------------- #
def create_all_files():
    import time
    import os
    import pandas as pd
    import multiprocessing    

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
        print("Creating file {}".format(dataSet["Output Name"]), flush=True)
        
        fileNames = os.listdir(sourcePath)
        argList = []
        for fileName in fileNames:
            argList.append([fileName, sourcePath, sourceType])

        pool = multiprocessing.Pool()
        result = pool.starmap_async(generate_features, argList)
        features = pd.concat(result.get())
        pool.close()
        pool.join()

        # Fill missing values with median of respective feature
        if sourceType == "Train":
            numFeatures = features.shape[1] - 2
        else:
            numFeatures = features.shape[1] - 1

        for i in range(1, numFeatures + 1):
            colName = "Feature" + str(i)
            features.loc[features[colName].isnull(), colName] = features[colName].dropna().median()

        features.to_csv(destFile, index = False)

    print("Creating file {}".format("train_all.csv"), flush=True)
    train_1 = pd.read_csv(rootDir + "train_1.csv")           
    train_2 = pd.read_csv(rootDir + "train_2.csv")           
    train_3 = pd.read_csv(rootDir + "train_3.csv")           
    train_all = pd.concat([train_1, train_2, train_3])
    train_all.to_csv(rootDir + "train_all.csv", index = False)

    print("Creating file {}".format("test_all.csv"), flush=True)
    test_1 = pd.read_csv(rootDir + "test_1.csv")           
    test_2 = pd.read_csv(rootDir + "test_2.csv")           
    test_3 = pd.read_csv(rootDir + "test_3.csv")           
    test_all = pd.concat([test_1, test_2, test_3])
    test_all.to_csv(rootDir + "test_all.csv", index = False)

    print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))

    return


# -------------------------------------------------------------------------------------- #
# Generate features 
# -------------------------------------------------------------------------------------- #
def generate_features(fileName, sourcePath, sourceType):

    import scipy.io as sio
    import pandas as pd
    #from IPython.core.debugger import Tracer; dbg_breakpoint = Tracer()
    
    samplingRate = 400
    timeWindows = [24000*time for time in range(11)] # 1 minute clips
    freqBands = [0.1, 4, 8, 12, 30, 70, 180]

    numChannels = 16
    numWindows = len(timeWindows)-1
    numBands = len(freqBands)-1
    numFeatures = numChannels * numWindows * numBands
    
    colNames = ["File"]
    if sourceType == "Train":
        colNames.append("Class")
    for i in range(1,numFeatures+1):
        colNames.append("Feature" + str(i))

    features = pd.DataFrame(columns = colNames)

    #dbg_breakpoint()
    try:
        fileContents = sio.loadmat(sourcePath + fileName, struct_as_record=False)
        fileContents = fileContents["dataStruct"][0,0]
        eegData = fileContents.data

        features.loc[1, "File"] = fileName
        if sourceType == "Train":
            features.loc[1, "Class"] = fileName.split(".")[0][-1]

#        for channel in range(numChannels):
#            for i in range(numWindows):
#                channelData = eegData[timeWindows[i]:timeWindows[i+1],channel]
#                freq, PSD = signal.periodogram(channelData, samplingRate)
#                totalPSD = sum(PSD)
#                for j in range(numBands):
#                    #dbg_breakpoint()
#                    freqFilter = np.logical_and(freq >= freqBands[j], freq < freqBands[j+1])
#                    featName = "Feature" + str(channel*numWindows*numBands + i*numBands + j + 1)
#                    features.loc[1, featName] = sum(PSD[freqFilter])/totalPSD
                    
    except ValueError:
        print("    Could not process file: " + fileName, flush=True)

    return features

# -------------------------------------------------------------------------------------- #
# Generate Shannon entropy features 
# -------------------------------------------------------------------------------------- #
def generate_shannon_entropy(eegData, timeWindows, freqBands, samplingRate):
    import scipy.signal as signal
    import numpy as np

    numEpochs = len(timeWindows)-1
    numFreqBands = len(freqBands)-1
    numChannels = eegData.shape[1]
    
    features = np.zeros(numChannels * numEpochs)
    
    for channel in range(numChannels):
        for i in range(numEpochs):
            epochData = eegData[timeWindows[i]:timeWindows[i+1],channel]
            freq, PSD = signal.periodogram(epochData, samplingRate)
            freqBinDensity = np.zeros(numFreqBands)
            for j in range(numFreqBands):
                freqFilter = np.logical_and(freq >= freqBands[j], freq < freqBands[j+1])
                freqBinDensity[j] = PSD[freqFilter].sum()/PSD.sum()
            
            features[channel*numEpochs + i] = -freqBinDensity.dot(np.log(freqBinDensity)) 
            
    return features
            

    
    
if __name__ == '__main__':
    create_all_files()







