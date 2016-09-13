# -------------------------------------------------------------------------------------- #
# Create submission file
# -------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"

dataSet1 = {"Train File": "train_1.csv", "Test File": "test_1.csv"}
dataSet2 = {"Train File": "train_2.csv", "Test File": "test_2.csv"}
dataSet3 = {"Train File": "train_3.csv", "Test File": "test_3.csv"}

dataSets = [dataSet1, dataSet2, dataSet3]
numFeatures = 384
lastFeature = "Feature" + str(numFeatures)
submission = pd.DataFrame(columns = ["File", "Class"])

for dataSet in dataSets:
    fileName = dataSet["Train File"]
    srcData = pd.read_csv(rootDir + fileName, index_col=0)
    for i in range(1, numFeatures + 1):
        feature = "Feature" + str(i)
        srcData.loc[srcData[feature].isnull(),feature] = srcData[feature].dropna().median()
    X_train = srcData.loc[:, "Feature1":lastFeature].values
    y_train = srcData.loc[:, "Class"].values

    fileName = dataSet["Test File"]
    srcData = pd.read_csv(rootDir + fileName, index_col=0)
    for i in range(1, numFeatures + 1):
        feature = "Feature" + str(i)
        srcData.loc[srcData[feature].isnull(),feature] = srcData[feature].dropna().median()
    X_test  = srcData.loc[:, "Feature1":lastFeature].values

    forest = RandomForestClassifier(n_estimators = 30)
    forest = forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    partialSub = pd.DataFrame(srcData["File"])
    partialSub["Class"] = y_pred
    
    submission = pd.concat([submission, partialSub])

submission.to_csv(rootDir + "submission.csv", index = False)


# -------------------------------------------------------------------------------------- #
# Cross Validation
# -------------------------------------------------------------------------------------- #
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier 

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"

numFeatures = 384
lastFeature = "Feature" + str(numFeatures)

fileName = "train_2.csv"
trainData = pd.read_csv(rootDir + fileName, index_col=0)

for i in range(1, numFeatures + 1):
    feature = "Feature" + str(i)
    trainData.loc[trainData[feature].isnull(),feature] = trainData[feature].dropna().median()


X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainData.loc[:, "Feature1":lastFeature].values,
                                                                     trainData.loc[:, "Class"].values, 
                                                                     test_size=0.3, 
                                                                     random_state=31415)

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

precision = cm[1,1]/(cm[1,1] + cm[0,1])
recall = cm[1,1]/(cm[1,1] + cm[1,0])

print(cm)
print("Precision: {} \nRecall: {}".format(precision, recall))



