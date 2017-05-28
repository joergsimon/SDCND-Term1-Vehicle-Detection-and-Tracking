# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

For details on the working of that project checkout the [report.md](./report.md)

---

The following files are important for the project:
* [report.md](./report.md) (report / project writeup)
* [report_files/](./report_files/) (folder with images and videos used for the report)
* [`Vehicle Detection and tracking.ipynb`](./Vehicle Detection and tracking.ipynb) (exploration of the algorithms based on single images and 6 frames)
* [test_images/](./test_images/) (folder with images used by the notebook)
* [`train_clf.py`](./train_clf.py) (the script used to train the classifiers)
* [`detect_and_track.py`](./detect_and_track.py) (the script used to annotate the video)
* [`helper/*.py`](./helper/) (python modules implementing the funcionality of the pipeline)
* [project_video.mp4](./project_video.mp4) (base video for later annotation)
* [result_video.mp4](./result_video.mp4) (final resulting video)

The video annotation is done with the script [`detect_and_track.py`](./detect_and_track.py). It has the values of the input video and the resulting video hardcoded, so if you want to change it, you have to change it in code.