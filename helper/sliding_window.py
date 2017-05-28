import cv2
import numpy as np
from .feature_extrection import extract_features_img

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    def cond_set(val, default):
        return val if val is not None else default

    x_start_stop[0] = cond_set(x_start_stop[0], 0)
    x_start_stop[1] = cond_set(x_start_stop[1], img.shape[1])

    y_start_stop[0] = cond_set(y_start_stop[0], 0)
    y_start_stop[1] = cond_set(y_start_stop[1], img.shape[0])

    span = (x_start_stop[1] - x_start_stop[0], y_start_stop[1] - y_start_stop[0])
    step = (int(xy_window[0] * xy_overlap[0]), int(xy_window[1] * xy_overlap[1]))
    shape = (np.int((span[0] - step[0]) / step[0]), np.int((span[1] - step[1]) / step[1]))
    num = shape[0] * shape[1]
    # print("debug: ", x_start_stop, y_start_stop, span, step, shape, num)
    window_list = []
    for ys in range(shape[1]):
        for xs in range(shape[0]):
            startx = xs * step[0] + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * step[1] + y_start_stop[0]
            endy = starty + xy_window[1]
            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def get_layers(image):
    layer1 = slide_window(image, y_start_stop=[380, None], xy_window=(192, 192))
    layer2 = slide_window(image, y_start_stop=[380, 650], xy_window=(128, 128))
    layer3 = slide_window(image, y_start_stop=[380, 600], xy_window=(96, 96))
    layer4 = slide_window(image, y_start_stop=[380, 550], x_start_stop=[100, 1200], xy_window=(64, 64))
    all_layers = layer1 + layer2 + layer3 + layer4
    return all_layers

def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel="ALL", spatial_feat=True,
                    hist_feat=True, hog_feat=True):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = extract_features_img(test_img, cspace=color_space, spatial_size=spatial_size,
                                 hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, use_spacial=spatial_feat,
                                 use_hist=hist_feat, use_hog=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def search_labeled_bboxes_images(img, labels, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel="ALL", spatial_feat=True,
                    hist_feat=True, hog_feat=True):
    on_car = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        test_img = cv2.resize(img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]], (64, 64))
        features = extract_features_img(test_img, cspace=color_space, spatial_size=spatial_size,
                                        hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, use_spacial=spatial_feat,
                                        use_hist=hist_feat, use_hog=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_car.append(bbox)
    return on_car