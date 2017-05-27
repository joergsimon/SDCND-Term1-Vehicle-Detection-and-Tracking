# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

[//]: # (Image References)
[image1]: ./report_files/example_vehicle.png "random vehicle"
[image2]: ./report_files/vehicle_hog.png "HOG of random vehicle"
[image3]: ./report_files/vehicle_hist_hls.png "Histogram of random vehicle"
[image4]: ./report_files/example_non_vehicle.png "random non vehicle"
[image5]: ./report_files/non_vehicle_hog.png "HOG of random non vehicle"
[image6]: ./report_files/non_vehicle_hist_hls.png "Histogram of random non vehicle"
[image7]: ./report_files/linSVC_overfittet_example.png "example of the overfitting of the LinearSVC"
[image8]: ./report_files/less-overfitting_example.png "example of the same image with a classifier less overfitting"
[image9]: ./report_files/stacked_grids.png "Search window stacked grids"
[video1]: ./report_files/must-be-overfitting.mp4 "example of the raw detected boxes of the overfitting of the LinearSVC"
[video2]: ./report_files/less-overfitting-clf.mp4 "example of a less overfitting model of a SVC with decision shape ovo and RBF Kernel"
## Basic organization in the project

The project is organised in two parts: One [python notebook](examples/example.ipynb) was used for exploration and tuning the algorithm for single images. Images in the report are generally taken from this notebook.

After experimenting with that two python scripts were created, one to train the classifier called `train_cld.py` who saves scaler and clf file as pickle and another called `detect_and_track.py` which loads the classifier and processes the video with the sliding window, classification, heatmap, labels pipeline explained later.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

An exploratory visualization of a vehicles and a non vehicles HOG can ge found in the [notebook](examples/example.ipynb) under feature visualization / hog features. Generally the parameters to tune are the number of orientations, the number of pixels per cell, and the cells per block. The last is a normalization parameter. I generally found that keeping all the default values but the cells per block works best. For the normalization I choose to normalise each cell individually. I found that works best when I tuned on LinearSVC at the beginning. Since this classifier was clearly overfitting something that result might not be that falid, but I sticked with it for now.

Here is the HOG of a random vehicle from the dataset:

![random choosen example vehicle][image1]

![HOG of the vehicle][image2]

Here is the HOG of a random non-vehicle from the dataset:

![random choosen example non-vehicle][image4]

![HOG of the non-vehicle][image5]

The results of that have been used in the extraction pipeline first for the classifier in `train_clf.py`
```python
vehicle_features = extract_features(vehicle_files, cspace="HLS", use_spacial=False, hist_range=(0,1), cell_per_block=1)
non_vehicle_features = extract_features(non_vehicle_files, cspace="HLS", use_spacial=False, hist_range=(0,1), cell_per_block=1)
```
and in the file `helper/feature_extrection.py` this is also employed in the notebook.

#### 2. Explain how you settled on your final choice of HOG parameters.

see above

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

My first approach was to train a LinearSVC as suggested in the course. Adding the raw pixel values in any color space only confused my classifier by about 5% so I dropped that. Histogram features at first had no real impact if I have them or not. I added some debug output when computing the features of the first example of the extraction, and found that the value range of the histogram to the hog features. Especially a SVM is pretty sensible to ranges of different features, and it does not really help if you later scale the whole vector of combined feature as the scaler will persist that changes. The maximum value of an value in the histogram is if the whole channel has the same intensity, f.e. in rgb all the same r. Then this r will get the value of `width*height` of the image, which is fixed to 32x32 in our case, which is 1024. Since most of the images will not have such a drastically color distribution I decided to simply divide the histogram feature fector by 512.0 to move it into a similar range of the hog features. That brang another 5%. For the HOG features experimentation showed that cells per block strangely gives good result, so I kept that. In the end I got a SVM who had about 97,4% accuracy on the test set.

The problem with that was, that the model clearly seemed to overfit on something wich is in the test and the training set which can be shown in the following image: 

![Example of an image where the SVM overfits][image7]

First I thought that will not be a problem, as the heatmap smoothing pipeline will take care of the problem, but this was not at all the case, detection of false positive reagon persisted trough the drive. To look what is happening I rendered a video with the raw pounding boxes from the sliding window search. The result shows that the model detects fals positives all the time in all kind of reagons in the image, so some overfitting must be present. Follow the link below if you are interested.

[video of the classifiers bounding box output / clearly overfitting][video1]

Since the heatmap could not take care of such a severe problem in the classifier I choose to explore the classification a bit more in depth. I decided to train several classifiers. Training was done with 5 fold cross validation to get the training accuracy. The following table shows the result of that training:

| Classifier         	| Parameters other then default |     Test accuracy 5 Fold avarage	    |
|:---------------------:|:-----------------------------:|:-------------------------------------:| 
| SVM (sklearn SVC)     | one vs. all (ovo), RBF kernel | 99.3%                     			|
| SVM (sklearn SVC) 	| RBF kernel    		        | 99.3%                     			|
| SVM (LinearSVC)    	|     				            | 97.2%                     			|
| Logistic Regression   |       				        | 97.4%                     			|
| Nearest Centroid   	|          				        | 90.6%                     			|
| Logistic Regression   | penalty=l1 (less features)    | 97.5%                     			|
| Stoch.Gradient Decent | loss=hinge, basicalls a LinearSVC    | 96.4%                     			|
| Decision Tree        	|          				        | 94.9%                     			|
| Random Forrest        | n_estimators=10               | 97.1%                     			|
| Ada Boost             | n_estimators=100              | 98.6%                     			|
| Gradient Boost        | n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0              | 98.3%                     			|

This is already interesting as it shows that several alternatives with similar ranges exist, and maybe one of them is not overfitting. The overfitting is likely on some features which diversify in this set only, because the LinearSVM had 97.4% on the validation set in the first try. Hoewever, maybe we get an idea if we now on the validation set compute the exact classification result and confusion matrix for each of the classifiers. This is shown in the next part:

---

SVM, one vs. all (ovo), RBF kernel

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 1.00      | 0.99   | 1.00     | 1793    | 
| 1.0        | 0.99      | 1.00   | 1.00     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 1.00      | 1.00   | 1.00     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1781      | 12     |
| 1.0        | 5         | 1754   |

---

SVM, RBF kernel had exactly the same result

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 1.00      | 0.99   | 1.00     | 1793    | 
| 1.0        | 0.99      | 1.00   | 1.00     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 1.00      | 1.00   | 1.00     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1781      | 12     |
| 1.0        | 5         | 1754   |

---

SVM (LinearSVC)

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.98      | 0.96   | 0.97     | 1793    | 
| 1.0        | 0.96      | 0.97   | 0.97     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.97      | 0.97   | 0.97     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1727      | 66     |
| 1.0        | 44        | 1715   |

---

Logistic Regression  

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.98      | 0.97   | 0.98     | 1793    | 
| 1.0        | 0.97      | 0.98   | 0.97     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.97      | 0.97   | 0.97     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1743      | 50     |
| 1.0        | 39        | 1720   |

---

Nearest Centroid   

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.91      | 0.88   | 0.90     | 1793    | 
| 1.0        | 0.89      | 0.92   | 0.90     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.90      | 0.90   | 0.90     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1585      | 208    |
| 1.0        | 149       | 1610   |

---

Logistic Regression

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.98      | 0.97   | 0.98     | 1793    | 
| 1.0        | 0.97      | 0.98   | 0.98     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.98      | 0.98   | 0.98     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1747      | 46     |
| 1.0        | 35        | 1724   |

---

Stoch.Gradient Decent

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.96      | 0.97   | 0.96     | 1793    | 
| 1.0        | 0.97      | 0.96   | 0.96     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.96      | 0.96   | 0.96     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1740      | 53     |
| 1.0        | 79        | 1680   |

---

Decision Tree 

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.95      | 0.96   | 0.95     | 1793    | 
| 1.0        | 0.96      | 0.95   | 0.95     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.95      | 0.95   | 0.95     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1716      | 77     |
| 1.0        | 90        | 1669   |

---

Random Forrest    

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.96      | 0.99   | 0.98     | 1793    | 
| 1.0        | 0.99      | 0.96   | 0.98     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.98      | 0.98   | 0.98     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1778      | 15     |
| 1.0        | 71        | 1688   |

---
   
Ada Boost 

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.99      | 0.99   | 0.99     | 1793    | 
| 1.0        | 0.99      | 0.98   | 0.99     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.99      | 0.99   | 0.99     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1775      | 18     |
| 1.0        | 27        | 1732   |

---

Gradient Boost

| Class      | precision | recall | f1-score | support |
|:----------:|:---------:|:------:|:--------:|:-------:|  
| 0.0        | 0.98      | 0.98   | 0.98     | 1793    | 
| 1.0        | 0.98      | 0.98   | 0.99     | 1759    |
|------------|-----------|--------|----------|---------|
| avg / total| 0.98      | 0.98   | 0.98     | 3552    |

Confusion Matrix

|            | 0.0       | 1.0    |
|:----------:|:---------:|:------:|
| 0.0        | 1766      | 27     |
| 1.0        | 27        | 1732   |

You can see that the Linear SVM does not have the highes precision or recall. So I tried to make a video with the SVM with a radial basis function kernel. There are still a lot of error but it is clearly less overfitting. All the models are pickelt so I can experiment with them when making the final video.

![less overfitting result of the classification][image8]

[Video of the detected bounding boxes of the SVM with RBF kernel][image8]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
