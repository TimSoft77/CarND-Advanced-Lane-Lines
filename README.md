**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/cal_original.PNG "Original"
[image2]: ./output_images/cal_undist.PNG "Undistorted"
[image3]: ./output_images/pipeline.png "Pipeline"

---
### Writeup / README

### Camera Calibration

This was done in `calibrate.py`.  The results were saved to pickle files to avoid repeating the calculations.

The process for camera calibration was:
1. A default object points array for a 9x6 checkerboard was prepared as `objp` (lines 7-9).
2. For each calibration image, the locations of the checkerboard corners were found and added to `imgpoints`. `objp` was also added to `objpoints` (lines 22-33).
3. The camera was calibrated and the calibration data was saved to pickle files (lines 35-41).

Sample original and undistorted images can be found below.

![alt text][image1] Original

![alt text][image2] Undistorted

---

### Pipeline (single images)

Image processing was done in `process_image.py`. The pipline is in the `process image` method (lines 42-271), which makes use of the `Line` class (lines 13-34) when processing multiple images (i.e., in a video).

The pipeline's steps are as follows:
#### 1. Produce an undistorted, transformed, thresholded binary image (lines 43-92)
1. Undistort the image, using the saved coefficients found by `calibrate.py`.
2. Convert to HLS and isolate the `s_channel`.
3. Convert the undistorted image to grayscale, take the sobel x gradient, and scale it (`scaled_sobel`).
4. Produce a binary 1-channel image (`combined_binary`) with ones wherever the `s_channel` or `scaled_sobel` values are within specified thresholds.  A binary 3-channel image (`color_binary`) is also produced with the acceptable `scaled_sobel` and `s_channel` values points on different channels, but it is for illustration purposes only.
5. Transform the perspective from the camera to top-down, producing the `binary_warped` image.
#### 2. Find the lane lines (lines 94-215)
1. Use the base points from the previous successful image if available; otherwise, the base points are the x values with the most pixels in the lower half of the image.
2. Identify all the pixels in a window near the base point.  If enough points are found, move the window left or right to line it up with the average pixel x-value.  Then move the window up, and find all the points in the new window.
3. Fit a 2nd order polynomial to all the pixels found in all the windows.
4. If the fitted polynomials for the left and right lanes stay roughly the same distance throughout the image, add them to the list `recent_xfitted`.  If not, increase the value of `num_errors`.
5. If `recent_xfitted` is not empty, the lane line is a weighted average of its values, giving greater weight to more recent values.
#### 3. Plot the lane on the undistorted image (lines 217-224)
1. This is done by drawing the lane lines on a blank image, inverse transforming it, and superimposing it on the original.
#### 4. Calculate the radii of curvature of the lane lines and position of the center of the car (lines 226-237)
1. Units are pixels.  The center of the car is the x-position in the image.
2. The number of recent consecutive unsuccessful attempts to find lane lines is also plotted.

The entire pipeline is illustrated in the image below.

![alt text][image3]

---

### Pipeline (video)

Here's a [link to my video result](./project_output.mp4)

---

### Discussion

Flaws of my pipeline:
1. It will fail whenever the distance between lanes is dropping.
2. It would completely fail on a painted road because the s-channel would always make its threshold.

Ideas for improvement:
1. If the lane lines are not both ok, check if either one is ok.  If so, make the other line a fixed difference from the acceptable one.
2. Make use of data about what the car is doing (e.g., speed, steering angle).
3. Give greater weight to pixels that meet both the s-channel or sobel x thresholds, or are at least near pixels of the other successful type.

