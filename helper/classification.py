def scale_features(features):
    from sklearn.preprocessing import StandardScaler
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(features)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(features)
    return scaled_X, X_scaler