import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog

# Helper functions:
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features

def bin_spatial(img, size=(32, 32)):
    imgcopy = img.copy()
    imgcopy = cv2.resize(imgcopy, size)
    features = imgcopy.ravel()
    return features

def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    image = mpimg.imread(car_list[0])
    data_dict["image_shape"] = image.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = image.dtype
    # Return data_dict
    return data_dict

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_per_block, cell_per_block), visualise=vis, feature_vector=feature_vec)
        return features

def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256),
                        orient=9, pix_per_cell=8, cell_per_block=2,
                        hog_channel="ALL", vis=False, feature_vec=True,
                        use_spacial=True, use_hist=True, use_hog=True):
    # Create a list to append feature vectors to
    features = []
    for image in imgs:
        img = mpimg.imread(image)
        imgcopy = img.copy()
        f = extract_features_img(imgcopy, cspace=cspace, spatial_size=spatial_size,
                                 hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, vis=vis, feature_vec=feature_vec,
                                 use_spacial=use_spacial, use_hist=use_hist, use_hog=use_hog)
        features.append(f)
    return features

def extract_features_img(img, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256),
                        orient=9, pix_per_cell=8, cell_per_block=2,
                        hog_channel="ALL", vis=False, feature_vec=True,
                        use_spacial=True, use_hist=True, use_hog=True):
    imgcopy = img.copy()
    imgcopy_fixed_res = cv2.resize(imgcopy, (32, 32))
    conversions = ['cv2.COLOR_RGB2HSV', 'cv2.COLOR_RGB2HLS', 'cv2.COLOR_RGB2LUV', 'cv2.COLOR_RGB2BGR']
    conv_pos = ["HSV", "HLS", "LUV", "BRG"]
    if cspace != 'RGB':
        conv = eval(conversions[conv_pos.index(cspace)])
        imgcopy = cv2.cvtColor(imgcopy, conv)
    f1 = bin_spatial(imgcopy, size=spatial_size) if use_spacial else []
    _,_,_,_,f2 = color_hist(imgcopy_fixed_res, nbins=hist_bins, bins_range=hist_range) if use_hist else []
    f3 = []
    if hog_channel == 'ALL':
        for channel in range(imgcopy_fixed_res.shape[2]):
            f3.append(get_hog_features(imgcopy_fixed_res[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        f3 = np.ravel(f3)
    else:
        f3 = get_hog_features(imgcopy_fixed_res[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    f = np.concatenate((f1, f2, f3))
    return f