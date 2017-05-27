import numpy as np
import pickle
from sklearn.svm import LinearSVC
import sklearn.linear_model as linear
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from helper.io import read_data_set
from helper.feature_extrection import data_look, extract_features
from helper.classification import scale_features, get_clf_candidates, fit_classifier

vehicle_files, non_vehicle_files = read_data_set("./data")
print(data_look(vehicle_files, non_vehicle_files))

vehicle_features = extract_features(vehicle_files, cspace="HLS", use_spacial=False, hist_range=(0,1), cell_per_block=1)
non_vehicle_features = extract_features(non_vehicle_files, cspace="HLS", use_spacial=False, hist_range=(0,1), cell_per_block=1)

print("vehicle_features shape:", vehicle_features[0].shape)

X = np.vstack((vehicle_features + non_vehicle_features)).astype(np.float64)
scaled_X, X_scaler = scale_features(X)

y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

clfs = get_clf_candidates()
fit_all = fit_classifier(X_train, y_train, clfs)
for name, clf, score in fit_all:
    pred = clf.predict(X_test)
    print('test clf ', name)
    print('---------------------------------------')
    print(classification_report(y_test, pred))
    print('')
    print('---------------------------------------')
    print(confusion_matrix(y_test, pred))

#with open('svc.p', 'wb') as f:
#    pickle.dump(clf, f)
#with open('scaler.p', 'wb') as f:
#    pickle.dump(X_scaler, f)