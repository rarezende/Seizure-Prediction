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

# Parameters: {'class_weight': {0: 0.001, 1: 0.999}, 'C': 0.001}
# ROC AUC Score: 0.685
# Total processing time: 6.14 minutes



# -------------------------------------------------------------------------------------- #
# Lasso Classifier     

from sklearn.linear_model import Lasso
classifier = Lasso()
param_grid = [
    {'alpha':[0.01, 0.02, 0.03]}
]

# Parameters: {'alpha': 0.01}
# ROC AUC Score: 0.675
# Total processing time: 0.22 minutes




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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/" 
fileName = "train_all.csv"
trainData = pd.read_csv(rootDir + fileName)

X = trainData.loc[:, "Feature1":].values
y = trainData.loc[:, "Class"].values

X = StandardScaler().fit_transform(X)

# -------- Change model here --------------
from sklearn.linear_model import Lasso
classifier = Lasso(alpha = 0.01)

scores = cross_val_score(classifier, X, y, cv=5, scoring = "roc_auc", n_jobs=-1)
print("ROC AUC: {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std()))



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
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(C=0.001, class_weight = {0: 0.001, 1: 0.999})
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 3000, class_weight = 'balanced')

X_train = StandardScaler().fit_transform(X_train)
classifier.fit(X_train, y_train)

X_test = StandardScaler().fit_transform(X_test)
y_pred = classifier.predict_proba(X_test)

submission = pd.DataFrame(testData["File"])
submission["Class"] = y_pred[:,1]
submission.to_csv(rootDir + "submission.csv", index = False)


# -------------------------------------------------------------------------------------- #
# Create submission file using ensemble
# -------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.preprocessing import StandardScaler

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"
trainData = pd.read_csv(rootDir + "train_all.csv")
testData = pd.read_csv(rootDir + "test_all.csv")

X_train = trainData.loc[:, "Feature1":].values
y_train = trainData.loc[:, "Class"].values

X_test  = testData.loc[:, "Feature1":].values

X_train = StandardScaler().fit_transform(X_train)

from sklearn.linear_model import Lasso
clsLasso = Lasso(alpha = 0.01)
clsLasso.fit(X_train, y_train)
y_predLasso = clsLasso.predict(X_train)

from sklearn.linear_model import LogisticRegression
clsLogit = LogisticRegression(C=0.001, class_weight={0: 0.001, 1: 0.999})
clsLogit.fit(X_train, y_train)
y_predLogit = clsLogit.predict_proba(X_train)[:,1]

X_ensemble = np.vstack((y_predLasso, y_predLogit)).T

from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators = 3000, learning_rate = 0.01)

classifier.fit(X_ensemble, y_train)

X_test = StandardScaler().fit_transform(X_test)
X_testLasso = clsLasso.predict(X_test)
X_testLogit = clsLogit.predict_proba(X_test)[:,1]
X_testEnsemble = np.vstack((X_testLasso, X_testLogit)).T

y_pred = classifier.predict_proba(X_testEnsemble)

submission = pd.DataFrame(testData["File"])
submission["Class"] = y_pred[:,1]
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
samplingRate = 400
i=0
channel = 0
epochData = eegData[timeWindows[i]:timeWindows[i+1],channel]




