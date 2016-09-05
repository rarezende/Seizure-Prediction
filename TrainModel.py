import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"

forest = RandomForestClassifier(n_estimators = 300)

# Data set 1
fileName = "train_1.csv"
trainData = pd.read_csv(rootDir + fileName, index_col=0)
X_train = trainData.loc[:, "Feature1":"Feature32"].values
y_train = trainData.loc[:, "Class"].values

fileName = "test_1.csv"
testData = pd.read_csv(rootDir + fileName, index_col=0)
X_test  = testData.loc[:, "Feature1":"Feature32"].values

forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
submission1 = pd.DataFrame(testData["File"])
submission1["Class"] = y_pred

# Data set 2
fileName = "train_2.csv"
trainData = pd.read_csv(rootDir + fileName, index_col=0)
X_train = trainData.loc[:, "Feature1":"Feature32"].values
y_train = trainData.loc[:, "Class"].values

fileName = "test_2.csv"
testData = pd.read_csv(rootDir + fileName, index_col=0)
X_test  = testData.loc[:, "Feature1":"Feature32"].values

forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
submission2 = pd.DataFrame(testData["File"])
submission2["Class"] = y_pred

# Data set 3
fileName = "train_3.csv"
trainData = pd.read_csv(rootDir + fileName, index_col=0)
X_train = trainData.loc[:, "Feature1":"Feature32"].values
y_train = trainData.loc[:, "Class"].values

fileName = "test_3.csv"
testData = pd.read_csv(rootDir + fileName, index_col=0)
X_test  = testData.loc[:, "Feature1":"Feature32"].values

forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
submission3 = pd.DataFrame(testData["File"])
submission3["Class"] = y_pred

submission = pd.concat([submission1, submission2, submission3])

submission.to_csv(rootDir + "submission.csv", index = False)

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainData.loc[:, "Feature1":"Feature32"].values, 
#                                                                     trainData.loc[:, "Class"].values, 
#                                                                     test_size=0.3, 
#                                                                     random_state=0)

#cm = confusion_matrix(y_test, y_pred)



