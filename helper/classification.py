def scale_features(features):
    from sklearn.preprocessing import StandardScaler
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(features)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(features)
    return scaled_X, X_scaler

# this code is taken from my master thesis and will appear there again ^_^
def get_clf_candidates():
    import sklearn.linear_model as lm
    import sklearn.neighbors.nearest_centroid as nc
    import sklearn.naive_bayes as nb
    import sklearn.ensemble as em
    from sklearn import svm
    from sklearn import tree
    classifiers = [("svc[ovo]", svm.SVC(decision_function_shape='ovo')),
                   ("svc", svm.SVC()),
                   ("lin svc", svm.LinearSVC()),
                   ("lr", lm.LogisticRegression()),
                   ("nn", nc.NearestCentroid()),
                   ("lr(l1)", lm.LogisticRegression(penalty='l1')),
                   ("sgd[hinge]", lm.SGDClassifier(loss="hinge", penalty="l2")),
                   ("decision tree", tree.DecisionTreeClassifier()),
                   ("random forrest", em.RandomForestClassifier(n_estimators=10)),
                   ("ada boost", em.AdaBoostClassifier(n_estimators=100)),
                   ("gradient boost",
                    em.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))]
    return classifiers

# this code is taken from my master thesis and will appear there again ^_^
def fit_classifier(values, labels, classifiers):
    import sklearn.model_selection as ms
    from sklearn.metrics import accuracy_score
    clf = None
    clf_n = None
    kf = ms.KFold(n_splits=5)
    result = []
    for clf_name, clf_candidate in classifiers:
        # decide: we should return the best model for each classifier here?
        # and do we pickle them?
        s = 0
        for train_index, test_index in kf.split(values):
            X_train = values[train_index, :]
            X_ct = values[test_index, :]
            y_train = labels[train_index]
            y_ct = labels[test_index]
            clf_candidate.fit(X_train, y_train)
            p = clf_candidate.predict(X_ct)
            s += accuracy_score(y_ct, p)
        s /= 5.0
        print(clf_name, ' has {:.3f}'.format(s))
        result.append((clf_name, clf_candidate, s))

    return result