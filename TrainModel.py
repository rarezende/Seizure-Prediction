import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier 

fileName = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/train1.csv"

trainData = pd.read_csv(fileName, index_col=0)

 X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainData.loc[:, "Feature1":"Feature32"].values, 
                                                                      trainData.loc[:, "Class"].values, 
                                                                      test_size=0.3, 
                                                                      random_state=0)

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators = 300)

# Fit the training data to the labels and create the decision trees
forest = forest.fit(X_train, y_train)

# Take the same decision trees and run it on the test data
output = forest.predict(X_test)



