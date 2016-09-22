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
    srcData = pd.read_csv(rootDir + fileName)
    for i in range(1, numFeatures + 1):
        feature = "Feature" + str(i)
        srcData.loc[srcData[feature].isnull(),feature] = srcData[feature].dropna().median()
    X_train = srcData.loc[:, "Feature1":lastFeature].values
    y_train = srcData.loc[:, "Class"].values

    fileName = dataSet["Test File"]
    srcData = pd.read_csv(rootDir + fileName)
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
import pandas as pd
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"
#rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/Archive/394 Features/"

fileName = "train_all.csv"
trainData = pd.read_csv(rootDir + fileName)

numFeatures = trainData.shape[1] - 2
lastFeature = "Feature" + str(numFeatures)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainData.loc[:, "Feature1":lastFeature].values,
                                                                     trainData.loc[:, "Class"].values, 
                                                                     test_size=0.2, 
                                                                     random_state=31415)


#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)

#LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
#                   intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', 
#                   max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1

#classifier = RandomForestClassifier(n_estimators = 30, class_weight= "balanced")
#classifier = svm.SVC(C=0.0000001, gamma=0.01, class_weight= "balanced")
#classifier = GradientBoostingClassifier(n_estimators= 10)

classifier = LogisticRegression(C=10, class_weight= "balanced")

X_train = StandardScaler().fit_transform(X_train)
classifier.fit(X_train, y_train)


# Test set predictions
X_test = StandardScaler().fit_transform(X_test)
y_pred = classifier.predict(X_test)
print("Confusion Matrix Test Set")
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

# Training set predictions
y_pred = classifier.predict(X_train)
print("----------------------------------------")
print("Confusion Matrix Train Set")
print(metrics.confusion_matrix(y_train, y_pred))
print(metrics.classification_report(y_train, y_pred))



