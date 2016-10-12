# -------------------------------------------------------------------------------------- #
# Parameter tuning
# -------------------------------------------------------------------------------------- #
def param_grid_search(classifier, param_grid):
    import time
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    startTime = time.time()
    
    rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"
    fileName = "train_all.csv"
    trainData = pd.read_csv(rootDir + fileName)

    X = trainData.loc[:, "Feature1":].values
    y = trainData.loc[:, "Class"].values

    X = StandardScaler().fit_transform(X)

    classifier = GridSearchCV(classifier, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    classifier.fit(X, y)

    print("Best parameters set found:")
    print("----------------------------")
    print("Parameters: " + str(classifier.best_params_))
    print("ROC AUC Score: {:0.3f}".format(classifier.best_score_))
    print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))
    print()
    print("Grid scores:")
    print("-----------------------------------------------")
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("{:0.3f} (+/-{:0.3f}) for {}".format(mean, std, params))



# -------------------------------------------------------------------------------------- #
# Parameter grids
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
# LogisticRegression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
param_grid = [
    {'C':[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], 
     'class_weight': ['balanced', {0:0.01, 1:0.99}, {0:0.005, 1:0.995}, {0:0.001, 1:0.999}]},
]

# Parameters: {'C': 0.003, 'class_weight': {0: 0.001, 1: 0.999}}
# ROC AUC Score: 0.66
# Total processing time: 1.45 minutes


# -------------------------------------------------------------------------------------- #
# RandomForestClassifier     

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
param_grid = [
    {'n_estimators':[1, 3, 10, 30, 100], 
     'max_depth': [None, 5, 10, 15, 30], 
     'max_leaf_nodes': [None, 5, 10, 15, 30]},
]

# Parameters: {'n_estimators': 100, 'max_leaf_nodes': None, 'max_depth': 10}
# ROC AUC Score: 0.60
# Total processing time: 3.56 minutes


# -------------------------------------------------------------------------------------- #
# Support Vector Machine

from sklearn import svm
classifier = svm.SVC()        
param_grid = [
  {'C': [5, 10, 15], 'kernel': ['rbf'], 'gamma': ['auto', 0.001, 0.003]},
  {'C': [5, 10, 15], 'kernel': ['poly'], 'degree': [2, 3, 5], 'gamma': ['auto', 0.001, 0.003]},
]

# Parameters: {'C': 10, 'kernel': 'rbf', 'gamma': 0.001}
# ROC AUC Score: 0.60
# Total processing time: 11.02 minutes

# -------------------------------------------------------------------------------------- #
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()
param_grid = [
    {'n_estimators':[1000, 3000], 
     'learning_rate': [0.003, 0.01], 
     'min_samples_split': [5, 0.01],
     'max_depth': [3, 4]},
]

# Parameters: {'max_depth': 3, 'learning_rate': 0.01, 'min_samples_split': 5, 'n_estimators': 3000}
# ROC AUC Score: 0.63
# Total processing time: 179.75 minutes




# -------------------------------------------------------------------------------------- #
# Sanity check for the tuned parameters
# -------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/" 
fileName = "train_all.csv"
trainData = pd.read_csv(rootDir + fileName)

X_train, X_test, y_train, y_test = train_test_split(trainData.loc[:, "Feature1":].values,
                                                    trainData.loc[:, "Class"].values, 
                                                    test_size=0.2, 
                                                    random_state=31415)

X = trainData.loc[:, "Feature1":].values
y = trainData.loc[:, "Class"].values

# -------- Change model here --------------
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.03, class_weight = {0: 0.01, 1: 0.99})

classifier = make_pipeline(StandardScaler(), classifier)
scores = cross_val_score(classifier, X, y, cv=5, scoring = "roc_auc", n_jobs=-1)
print("ROC AUC: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std()))



# -------------------------------------------------------------------------------------- #
# Create submission file
# -------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.preprocessing import StandardScaler

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"
trainData = pd.read_csv(rootDir + "train_all.csv")
testData = pd.read_csv(rootDir + "test_all.csv")

X_train = trainData.loc[:, "Feature1":].values
y_train = trainData.loc[:, "Class"].values

X_test  = testData.loc[:, "Feature1":].values

# -------- Change model here --------------
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.03, class_weight = {0: 0.01, 1: 0.99})

X_train = StandardScaler().fit_transform(X_train)
classifier.fit(X_train, y_train)

X_test = StandardScaler().fit_transform(X_test)
y_pred = classifier.predict(X_test)

submission = pd.DataFrame(testData["File"])
submission["Class"] = y_pred
submission.to_csv(rootDir + "submission.csv", index = False)



# -------------------------------------------------------------------------------------- #
# Load epochData for testing
# -------------------------------------------------------------------------------------- #
import os
import scipy.io as sio

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"
sourcePath = rootDir + "Data1/train_1/"
fileNames = os.listdir(sourcePath)
fileName = fileNames[0]
fileContents = sio.loadmat(sourcePath + fileName, struct_as_record=False)
fileContents = fileContents["dataStruct"][0,0]
eegData = fileContents.data
timeWindows = [24000*time for time in range(11)] # 1 minute clips
i=0
channel = 0
epochData = eegData[timeWindows[i]:timeWindows[i+1],channel]




