**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image2_1]: ./examples/car_1.png
[image2_2]: ./examples/car_2.png
[image2_3]: ./examples/car_3.png
[image2_4]: ./examples/car_4.png
[image2_5]: ./examples/car_5.png
[image2_6]: ./examples/car_6.png
[image2_7]: ./examples/notcar_1.png
[image2_8]: ./examples/notcar_2.png
[image2_9]: ./examples/notcar_3.png
[image2_10]: ./examples/notcar_4.png
[image2_11]: ./examples/notcar_5.png
[image2_12]: ./examples/notcar_6.png
[image3]: ./examples/sliding_windows_1.png
[image4]: ./examples/sliding_window.jpg
[image4_1]: ./examples/sliding_window_1.png
[image4_2]: ./examples/sliding_window_2.png
[image4_3]: ./examples/sliding_window_3.png
[image4_4]: ./examples/sliding_window_4.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image6_1]: ./examples/labels_map_1.png
[image6_2]: ./examples/labels_map_2.png
[image6_3]: ./examples/labels_map_3.png
[image6_4]: ./examples/labels_map_4.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (P5.ipynb).  

I started by reading in all the `vehicle` and `non-vehicle` images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using color space and HOG parameters of `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2_1]
![alt text][image2_2]
![alt text][image2_3]
![alt text][image2_4]
![alt text][image2_5]
![alt text][image2_6]
![alt text][image2_7]
![alt text][image2_8]
![alt text][image2_9]
![alt text][image2_10]
![alt text][image2_11]
![alt text][image2_12]

#### 2. Explain how you settled on your final choice of HOG parameters.

I wrote a trainer to train classfifier with differet HOG parameters: (trainer.ipynb)
* color_space: YCrCb HSV HLS YUV LUV GRAY
* orient: 8 9
* pix_per_cell: 8 16
* cell_per_block: 1 2
* hog_channel: 0 1 2 'ALL'
* feature comination: hog_feat only, hog_feat+hist_feat+spatial_feat

First round, after comparing all the models accuracy, I found:
* pix_per_cell: 16 is better than 8
* cell_per_block = 2 is better than 1
* orient = 8 or 9 is a good choice, 8 is a little better sometimes
* pix_per_cell = 16, cell_per_block = 2, orient = 8 or 9 is a good choice of hog parameters

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Second round, comparing all the models performace on test images, I found:
* hog_feat only is better than hog_feat+hist_feat+spatial_feat
* HSV HLS GRAY is better than YCrCb YUV LUV RGB
* ALL hog_channel is better than single hog_channel

Third round, I tried parameters: pix_per_cell = 16, cell_per_block = 2, orient = 8, HSV hog_feat with in detail:
* hog_channel H is not very good to classify cars and not cars
* hog_channel S is good
* hog_channel V is best

Finally, I decided to use hog_channel_S + hog_channel_V with parameters pix_per_cell = 16, cell_per_block = 2, orient = 8

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used a combination of window_size, y_axis and overlap:
* window_size: (64,64) - red, (96,96) - green, (128,128) - blue, (160,160) - white
* y_axis: [368,624], [368,624], [368,624, [368,624]
* overlap: (0.5, 0.5)

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried different ways and many models on test images:
* ConvNet neural network model has a high accuracy, but can not recognize part of a car very well
* SVM+Hog models can recognize part of a car better than cnn model, but always recognize false result

Finally, I chose SVM+Hog model

![alt text][image4_1]
![alt text][image4_2]
![alt text][image4_3]
![alt text][image4_4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

* I recorded the positions of positive detections in each frame of the video.
* From the positive detections I created a heatmap, I defined a queue `heatmaps` to store latest 20 frames heatmaps
* Then thresholded `heatmaps` with `probability` to identify vehicle positions, `probability` means how many times appear in last 20 heatmaps.  
* I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
* I then assumed each blob corresponded to a vehicle.  
* I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is example of six frames and their corresponding heatmaps:

![alt text][image5]

### Here is example of the output heatmap and lables:
![alt text][image4_1]
![alt text][image6_1]
![alt text][image4_2]
![alt text][image6_2]
![alt text][image4_3]
![alt text][image6_3]
![alt text][image4_4]
![alt text][image6_4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Flase positives were big troubles. I always got false on different models even with a high accuracy (>0.98). I augmented cars data to 9000+ and non-vehicle data to 50000+ to minimize false positives.

2. Low efficiency on video frame. I only got 3~4 FPS on search window hog features algorithm. If overlap parameter is bigger(=0.75), the efficiency would be worse. I found YOLO is a good choice, seems very cool. Maybe I can try YOLO futhur on this project. RCNN and Faster RCNN maybe a better solution to this project.
