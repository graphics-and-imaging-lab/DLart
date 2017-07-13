from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from scipy import misc
import numpy as np
from collections import defaultdict
from sklearn.externals import joblib
import utils
import sys


def fineTunning(feat_crossv, classes_crossv):
    # Set the parameters by cross-validation
    tuned_parameters = \
        [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'decision_function_shape': ["ovr", "ovo"]},
         {'kernel': ['poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10,
                                                           100, 1000], 'decision_function_shape': ["ovr", "ovo"]},
            {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10,
                                                                 100, 1000], 'decision_function_shape': ["ovr", "ovo"]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000],
             'decision_function_shape': ["ovr", "ovo"]}
         ]
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=3,
                           scoring='%s_weighted' % score)
        clf.fit(feat_crossv, classes_crossv)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.

dataPath = '/media/mlagunas/a0148b08-dc3a-4a39-aee5-d77ee690f196/TFG'
net = "vgg19"
name = "curated"  # noisy23 / noisy/ curated
typ = "curated"
features = dataPath + "/h5/" + typ + "/features/" + \
    net + "/" + name + "_" + net + "_"
cl = "../../data/paths/" + typ + "/" + name + "_paths"

feat = utils.getFeatures(features + "42.h5", "features")
feat_train = utils.getFeatures(features + "train_42.h5", "features")
feat_crossv = feat_train[:int(len(feat_train) * 0.2)]
feat_test = utils.getFeatures(features + "test_42.h5", "features")
# Load classes
classes, path = utils.getClasses(cl + ".txt")
classes_train, path_train = utils.getClasses(cl + "_train.txt")
classes_crossv, path_crossv = classes_train[
    :int(len(classes_train) * 0.2)], path_train[:int(len(path_train) * 0.2)]
classes_test, path_test = utils.getClasses(cl + "_test.txt")


# Get the best parameters for the dataset
fineTunning(feat_crossv, classes_crossv)

# Create the SVM with the given parameters
clf = svm.SVC(kernel='rbf', C=1, decision_function_shape='ovr',
              gamma=0.0001, probability=True)
clf = svm.SVC(kernel='sigmoid', C=10, decision_function_shape='ovr',
              gamma=0.0001, probability=True)
clf.fit(feat_train, classes_train)
# Save the model
joblib.dump(clf, dataPath + '/test/SVM/SVM_vgg19.pkl')
