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
[image10]: ./report_files/found_boxes_6frames.png "found boxes by random forrest for 6 frames"
[image11]: ./report_files/direct_headmap.png "direct heatmap from the bounding boxes"
[image12]: ./report_files/heatmap_w_history_without_threshold.png "avaraged heatmap history for 6 frames"
[image13]: ./report_files/heatmap_w_history_thresh.png "thresholded avaraged heatmap for 6 frames"
[image14]: ./report_files/bbox_6images.png "found final bounding box in 6 frames"
[image15]: ./report_files/label_6images.png "label algorithm output for 6 frames"
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

This is already interesting as it shows that several alternatives with similar ranges exist, and maybe one of them is not overfitting. The overfitting is likely on some features which diversify in this set only, because the LinearSVM had 97.4% on the validation set in the first try. Hoewever, maybe we get an idea if we now on the validation set compute the exact classification result and confusion matrix for each of the classifiers. The end of the script in `train_clf.py` of the notebook computes detailed results. This is shown in the next part:

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

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The test dataset has images of whole vehicles or non vehicles. While driving the car slides towards the horizon changing size in the picture. To be able to catch the whole process therefor a algorithm must be able to capture the car at most of these positions. However, having all kinds of resolutions over the complete window is computationally a bad idea.

No vehicles will appear in the upper part of the images, so we can ignore the search space there. Small vehicles will appear more in the middle of the image as this is where the horizon goes, and not at the bottom of the image, there only larger cars can be found. To capture this effect of the perspective grids with different resolutions (192x192, 128x128, 96x96, 64x64) all with 50% overlapp are stacked over each other. All start approximately at the horizon, but the high gets smaller the smaller the grid is. The next image visualizes this strategy.

![Sliding window search grids][image9]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Taking 6 frames out of the window and into the notebook allowed me to explore the effect of classifiers pretty well. it was interesting to see that the general numbers reportet in section one did not have that much of an influence. In the end NearestCentroid, Logistic Regression, Random Forrest and Gradient Boost had the most meaningful per frame performance in this 6 frames, something which I would never have guessed with the results above. Also at first I tried just rendering the whole pipeline. May things happen there so the influence of the classifier was not that clear, having these 6 frames really helped me in that regard. For the final video I then did choose the Random Forrest classifier. While beeing an esamble classifier it still was not that slow and had the best detection based on visual inspection.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap. I kept a queue of the raw heatmaps over the last 10 frames, and computed the avarage of them. I then thresholded that avaraged map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their windows found by random forrest:

![windows found by random forrest][image10]

### Here are six frames and their corresponding heatmaps:

This images shows the directly computed heatmaps from each frame

![heatmap direct][image11]

10 frames are kept and avaraged

![heatmap avaraged over history][image12]

this keeps a memory of false positives, but also gives them a way smaller weight after the araraging. So we threshold with 0.9 to get them out.

![thresholded avaraged heatmap][image13]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![output of the labels algorithm for this 6 frames][image15]

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![final bounding box][image14]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For the winning video I used an essamble classifier (Ramdon Forrest). While that works really well, it is not as fast as a linear SVM. If that should be used for real time detection strong hardware would be needed increasing the costs. I did also not optimize the computation of HOG features as adviced in the course as this would have led to a too large change in code given the time. This would also speed up the computation a lot and help. So overall performance is an issue of my approach.

Additionally to that I think that the strategy of the sliding window could be optimized by making it more dynamic. F.e. you could train a classifier for a quick rought detection of a vehicle in a larger space and if that is detected refine the grid dynamically. Or moving the grid around maybe.

When two vehicles cross each other, the current algorithm merges the two vehicles to one because of the heatmap. Remembering the number of vehicles and move their centroids would help a lot here (unless one car takes a turn when he is shadowed by the other car, then the could would stay wrong for the reminder of the first car beeing present). Another option would be to not use the label algorithm but maybe some other clustering method to see if it finds better clusters.
