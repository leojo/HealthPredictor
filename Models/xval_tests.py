import csv
import numpy as np
from random import shuffle
from sklearn import pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import *
from sklearn.gaussian_process import *
from sklearn.neural_network import *
from sklearn.preprocessing import *
from sklearn.decomposition import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

from VotingRegressor import VotingRegressor
from LooRegressor import LooRegressor
from Features.extract_features import *

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.asarray([int(x[0]) for x in targets])

data = np.asarray(extractHistograms("../data/set_train",4500,45,9))
print "Shape of data:"
print np.array(data).shape

print "Estimating error:"

models = {
	"AdaBoost (select1400best)" : pipeline.make_pipeline(
		SelectKBest(k=1400),
		AdaBoostClassifier()
	),
	"AdaBoost (PCA 1400 - whiten)" : pipeline.make_pipeline(
		PCA(n_components=1400,whiten=True),
		AdaBoostClassifier()
	),
	"AdaBoost (PCA 1400)" : pipeline.make_pipeline(
		PCA(n_components=1400),
		AdaBoostClassifier()
	),
	"AdaBoost (scaler - select 1400 - PCA 100)" : pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=1400),
		PCA(n_components=100),
		AdaBoostClassifier()
	),
	"Nearest neighbours-3 (PCA 10)": pipeline.make_pipeline(
		PCA(n_components=10),
    	KNeighborsClassifier(3)
	),
	"Nearest neighbours-2 (PCA 10)": pipeline.make_pipeline(
		PCA(n_components=10),
    	KNeighborsClassifier(3)
	),
	"Nearest neighbours-1 (PCA 10)": pipeline.make_pipeline(
		PCA(n_components=10),
    	KNeighborsClassifier(3)
	),
	"Nearest neighbours-4 (PCA 10)": pipeline.make_pipeline(
		PCA(n_components=10),
    	KNeighborsClassifier(3)
	)
}

for key, model in sorted(models.items()):
	scores = cross_val_score(model, data, targets, cv=10, scoring='neg_log_loss', n_jobs=-1)
	print "score: %0.2f (+/- %0.2f) [%s]" % (-scores.mean(), scores.std(),key)


#hist_targ = zip(data,targets)
#shuffle(hist_targ)
#n_test_data = 30
#n_tests = 100
#errors = {}
#for i in range(n_tests):
#	X_shuffled, y_shuffled = zip(*hist_targ)
#	X_train = X_shuffled[:-n_test_data]
#	y_train = y_shuffled[:-n_test_data]
#	X_test = X_shuffled[-n_test_data:]
#	y_test = y_shuffled[-n_test_data:]
#	for key, model in models.items():
#		model.fit(X_train,y_train)
#		prediction = np.array(model.predict(X_test))
#		truth = np.array(y_test)
#		difference = np.absolute(prediction-truth)
#		average_error = sum(difference)
#		if key in errors:
#			errors[key] += (average_error*1.)/n_tests
#		else:
#			errors[key] = (average_error*1.)/n_tests
#
#for key, error in sorted(errors.items()):
#	print "Average number of errors: %f [%s]"%(error,key)

