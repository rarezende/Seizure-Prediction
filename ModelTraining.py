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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score

rootDir = "C:/Users/Rodrigo/Documents/Data Science/Seizure-Prediction/Data/"

fileName = "train_all.csv"
trainData = pd.read_csv(rootDir + fileName)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainData.loc[:, "Feature1":].values,
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

X = trainData.loc[:, "Feature1":].values
y = trainData.loc[:, "Class"].values

classifier = LogisticRegression(C=20, class_weight="balanced")
classifier = make_pipeline(StandardScaler(), classifier)
scores = cross_val_score(classifier, X, y, cv=5, scoring = "roc_auc")
print("ROC AUC: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std()))


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


