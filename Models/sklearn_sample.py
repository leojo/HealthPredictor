import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from Features.extract_features import *
import csv
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(base_estimator=SVC(kernel="linear", C=0.025, probability=True)),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

'''
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
'''
'''datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]'''

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)
targets = np.asarray([int(x[0]) for x in targets])           

'''
#histograms
histoData = np.asarray(extractHistograms("../data/set_train", 4500, 45, 16))
pcaHisto = PCA(n_components=2)
pcaHistoData = pcaHisto.fit_transform(histoData)
myDatasetHisto = (pcaHistoData, targets)


# zones
zoneData = np.asarray(extractZoneAverages("../data/set_train", 8))
pcaZone = PCA(n_components=2)
pcaZoneData = pcaZone.fit_transform(zoneData)
myDatasetZone = (pcaZoneData, targets)

# zones 
blackData = np.asarray(extractBlackzones("../data/set_train", 8))
pcaBlack = PCA(n_components=2)
pcaBlackData = pcaBlack.fit_transform(blackData)
myDatasetBlack = (pcaBlackData, targets)

blackData2 = np.asarray(extractBlackzones("../data/set_train", 8))
blackData2 = blackData2*blackData2
pcaBlack2 = PCA(n_components=2)
pcaBlackData2 = pcaBlack2.fit_transform(blackData2)
myDatasetBlack2 = (pcaBlackData2, targets)

blackData3 = np.asarray(extractBlackzones("../data/set_train", 8))
print blackData3
blackData3 = np.log10((blackData3+np.asarray([[1]*len(blackData3[0])]*len(blackData3))))
print blackData3
pcaBlack3 = PCA(n_components=2)
pcaBlackData3 = pcaBlack3.fit_transform(blackData3)
myDatasetBlack3 = (pcaBlackData3, targets)'''

# zones
fullData = np.asarray(extractFlipSim("../data/set_train"),scale = 0.1)
pcaFull = PCA(n_components=2)
pcaFullData = pcaFull.fit_transform(fullData)
myDatasetFull = (pcaFullData, targets)
'''
fourFifthsData = np.asarray(extractFlipSim("../data/set_train", scale = 0.2))
pcaFourFifths = PCA(n_components=2)
pcaFourFifthsData = pcaFourFifths.fit_transform(fourFifthsData)
myDatasetFourFifths = (pcaFourFifthsData, targets)

threeFifthsData = np.asarray(extractFlipSim("../data/set_train", scale = 0.15))
pcaThreeFifths = PCA(n_components=2)
pcaThreeFifthsData = pcaThreeFifths.fit_transform(threeFifthsData)
myDatasetThreeFifths = (pcaThreeFifthsData, targets)

twoFifthsData = np.asarray(extractFlipSim("../data/set_train", scale = 0.1))
pcaTwoFifths = PCA(n_components=2)
pcaTwoFifthsData = pcaTwoFifths.fit_transform(twoFifthsData)
myDatasetTwoFifths = (pcaTwoFifthsData, targets)

oneFifthData = np.asarray(extractFlipSim("../data/set_train", scale = 0.05))
pcaOneFifth = PCA(n_components=2)
pcaOneFifthData = pcaOneFifth.fit_transform(oneFifthData)
myDatasetOneFifth = (pcaOneFifthData, targets)
print oneFifthData.shape'''
datasets = [myDatasetFull]#, myDatasetFourFifths, myDatasetThreeFifths, myDatasetTwoFifths, myDatasetOneFifth]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    '''X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
'''
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    '''# Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)'''
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    # and testing points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        score = cross_val_score(clf, X, y, cv=10, scoring='neg_log_loss', n_jobs=-1)#clf.score(X_test, y_test)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
        # and testing points
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % -score.mean()).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()