from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import seaborn as sn
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC
import utils
import h5py as h5


def accuracy_top_n(n_results, curated_synset, probs, classes_test):
    zipped = [sorted(zip(i, curated_synset), reverse=True) for i in probs]
    zipped = np.array(zipped)
    pred_classes = zipped[:, :n_results, 1]
    pred_probs = zipped[:, :n_results, 0]

    top_n = np.array([pred_classes[i] == classes_test[i]
                      for i in range(len(pred_classes))])
    x, y = np.where(top_n)

    pred_index = np.ones(x[-1] + 1) * (-1)
    for i in range(len(x)):
        pred_index[x[i]] = pred_index[x[i]] * y[i] * (-1)

    pred_index = pred_index.astype(int)
    classes_test = np.array(classes_test)

    acc_class = np.zeros(len(curated_synset))
    for i in range(len(curated_synset)):
        c = curated_synset[i]
        pred_c = pred_index[classes_test == c]
        acc_class[i] = float(len(pred_c[pred_c != -1])) / float(len(pred_c))
    print ('global', float(
        len(pred_index[pred_index != -1])) / float(len(pred_index)))
    return zip(curated_synset, acc_class)

curated_synset_path = "/home/mlagunas/Bproject/DLart/data/data_utils/synset_curated.txt"
test = ""
dataPath = '/media/mlagunas/a0148b08-dc3a-4a39-aee5-d77ee690f196/TFG/test'
net = "vgg19"
name = "curated"  # noisy23 / noisy/ curated
dataset = "curated"
clf = joblib.load(dataPath + "/SVM/SVM_vgg19.pkl")

features = dataPath + "/h5/" + dataset + "/features/" + \
    net + "/" + name + "_" + net + "_"
cl = "../../data/paths/" + dataset + "/" + name + "_paths"

# Get features
feat_test = utils.getFeatures(features + "test_42.h5", "features")
classes_test, path_test = utils.getClasses(cl + "_test.txt")

with open(curated_synset_path) as f:
    curated_synset = f.read().splitlines()

######################################################
## Getting the accuracy
probs = clf.predict_proba(feat_test)
probs = probs.tolist()
accuracy_top_n(5, curated_synset, probs, classes_test)
accuracy_top_n(1, curated_synset, probs, classes_test)

y_true, y_pred = classes_test, clf.predict(feat_test)

%matplotlib
cnf_matrix = confusion_matrix(y_true, y_pred, labels=curated_synset)
df_cm = pd.DataFrame(cnf_matrix, index=[i for i in curated_synset],
                     columns=[i for i in curated_synset])
df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
sn.set(font_scale=1.3)
sn.heatmap(df_cm,
           cmap="Blues",
           linewidths=.5)

# This sets the yticks "upright" with 0, as opposed to sideways with 90.
plt.yticks(rotation=0)
plt.xticks(rotation=-65)
sn.plt.tight_layout()
sn.plt.show()
