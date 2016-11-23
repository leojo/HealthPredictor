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

from Features.extract_features import *

# Get the targets
with open('../data/targets.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = list(reader)

targets = np.asarray([int(x[0]) for x in targets])

data = np.asarray(extractHierarchicalClusters("../data/set_train",50,scale=0.1))
print "Shape of data:"
print np.array(data).shape

print "Estimating error:"

models = {
	"SVC" : pipeline.make_pipeline(
		StandardScaler(),
		SelectKBest(k=40),
		SVC(kernel='poly', probability=True)
	),
	"RandomForestClassifier" : pipeline.make_pipeline(
		StandardScaler(),
		RandomForestClassifier(max_depth=20,n_estimators=600,max_features=3)
	)
}

for key, model in sorted(models.items()):
	print(key)
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

