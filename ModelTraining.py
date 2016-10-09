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
    print("ROC AUC Score: {:0.2f}".format(classifier.best_score_))
    print("Total processing time: {:.2f} minutes".format((time.time()-startTime)/60))
    print()
    print("Grid scores:")
    print("-----------------------------------------------")
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("{:0.2f} (+/-{:0.2f}) for {}".format(mean, std, params))



# -------------------------------------------------------------------------------------- #
# Parameter grids
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
# LogisticRegression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
param_grid = [
    {'C':[0.03, 0.1, 0.3, 1, 3, 10, 30, 100], 
     'fit_intercept': [True, False], 
     'class_weight': [None, 'balanced', {0:0.05, 1:0.95}, {0:0.95, 1:0.05}]},
]

# Parameters: {'fit_intercept': True, 'class_weight': {0: 0.05, 1: 0.95}, 'C': 0.3}
# Score: 0.62


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
  {'C': [0.3, 1, 3], 'kernel': ['linear']},
  {'C': [0.3, 1, 3], 'kernel': ['rbf'], 'gamma': ['auto', 0.1, 0.01]},
  {'C': [0.3, 1, 3], 'kernel': ['poly'], 'degree': [2, 3, 5], 'gamma': ['auto', 0.1, 0.01]},
]




# -------------------------------------------------------------------------------------- #
# Sanity check for the tuned parameters
# -------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/" 
fileName = "train_all.csv"
trainData = pd.read_csv(rootDir + fileName)

X_train, X_test, y_train, y_test = train_test_split(trainData.loc[:, "Feature1":].values,
                                                    trainData.loc[:, "Class"].values, 
                                                    test_size=0.2, 
                                                    random_state=31415)

X = trainData.loc[:, "Feature1":].values
y = trainData.loc[:, "Class"].values

classifier = LogisticRegression(C=0.3, class_weight="balanced")

classifier = make_pipeline(StandardScaler(), classifier)
scores = cross_val_score(classifier, X, y, cv=5, scoring = "roc_auc")
print("ROC AUC: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std()))



# -------------------------------------------------------------------------------------- #
# Create submission file
# -------------------------------------------------------------------------------------- #
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"
trainData = pd.read_csv(rootDir + "train_all.csv")
testData = pd.read_csv(rootDir + "test_all.csv")

X_train = trainData.loc[:, "Feature1":].values
y_train = trainData.loc[:, "Class"].values

X_test  = testData.loc[:, "Feature1":].values

classifier = LogisticRegression(C=0.01, class_weight= "balanced")

X_train = StandardScaler().fit_transform(X_train)
classifier.fit(X_train, y_train)

X_test = StandardScaler().fit_transform(X_test)
y_pred = classifier.predict(X_test)

submission = pd.DataFrame(testData["File"])
submission["Class"] = y_pred
submission.to_csv(rootDir + "submission.csv", index = False)


