import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from helper.io import read_data_set
from helper.feature_extrection import data_look, extract_features
from helper.classification import scale_features

vehicle_files, non_vehicle_files = read_data_set("./data")
print(data_look(vehicle_files, non_vehicle_files))

vehicle_features = extract_features(vehicle_files, cspace="HSV", use_spacial=False)
non_vehicle_features = extract_features(non_vehicle_files, cspace="HSV", use_spacial=False)

print("vehicle_features shape:", vehicle_features[0].shape)

X = np.vstack((vehicle_features + non_vehicle_features)).astype(np.float64)
scaled_X, X_scaler = scale_features(X)

y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
svc = LinearSVC()
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

with open('svc.p', 'wb') as f:
    pickle.dump(svc, f)
with open('scaler.p', 'wb') as f:
    pickle.dump(X_scaler, f)