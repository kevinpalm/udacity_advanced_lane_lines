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

[image1]: ./output_images/corrected_calibration10.jpg "Calibration"
[image2]: ./output_images/undistorted.jpg "Undistorted"
[image3]: ./output_images/color_threshold.jpg "Binary Example"
[image4]: ./output_images/straight_lines.jpg "Straight lines"
[image5]: ./output_images/histogram_fit.jpg "Histogram Fit Visual"
[image6]: ./output_images/margin_fit.jpg "Margin Fit Visual"
[image7]: ./test_images/processedtest2.jpg "Output"
[video1]: ./processed_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 10 through 61 of the file called `lane_find.py`.

I'm using images of chessboards to calibrate my camera. I started by creating empty placeholder arrays to append object points and image points into. Object points correspond to the hypothetical (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `imobjp` is just a replicated array of coordinates, and `objp` is the array that will be appended with a copy of `imobjp` every time I successfully detect all chessboard corners in a test image.  `imgp` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output arrays `objp` and `imgp` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  Here's an example of what one of the calibration images looks like with the distortion correction applied:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
All imaged are corrected for distortion by using the `cv2.undistort()`. Here's an example of one of the test images with undistortion applied:

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (steps at lines 84 through 134 in `lane_find.py`). For the color channel, I used the saturation channel from HSV color scheme. For the gradient threshold, I created yellow and white masks for filtering out likely lane lines, then applied those filters to a grayscale image, then caluculated the sorbel x derivitave, applied absolute value, and scaled to between 0-255. On both channels, the minimum threshold is equal to 3.4 standard deviations above the mean value of the bottom half of the image. The maximum threshold is 255 for each. Here's an example of my output for this step.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `top_down_perspective_transform()`, which appears in line 65 in the file `lane_find.py`.  The `top_down_perspective_transform()` function takes as inputs an image (`image`). It then uses earlier defined transformations from `cv2.getPerspectiveTransform` to convert to a top down perspective. The transformation is based off hardcoded, ratio-based source and destination points, which are:
```
sourcepoints = np.float32([[int(img.shape[1]*217/384), int(img.shape[0]*28/44)],
                           [int(img.shape[1]), int(img.shape[0])],
                           [0, int(img.shape[0])],
                           [int(img.shape[1]*167/384), int(img.shape[0]*28/44)]])
destinationpoints = np.float32([[img.shape[1], 0],
                                [img.shape[1], img.shape[0]],
                                [0, img.shape[0]],
                                [0, 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 723, 458      | 1280, 0       | 
| 1280, 720     | 1280, 720     |
| 0, 720        | 0, 720        |
| 557, 458      | 0, 0          |

I verified that my perspective transform was working as expected by transforming images of straight roads and verifing that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

My code uses two methods for identifying lane-line pixels. The first method is to use a histogram to find where the most "active" pixels are located, in a series of "windows" as it works it's way up the frame. My code uses 9 windows, searches a margin of 100 pixels, and - while working its way up - it requires at least 50 new active pixels to slide the window.

![alt text][image5]

The other method my code uses for identifying lane-line pixels is to search along the margin of previous lane lines. It prefers to use this method whenever it's working through a video and prior lane lines are available. The code for this secion is from line 283 to 312 in `lane_find.py`. When it uses this method, however, it has to pass an extra check to make sure that the cuvature of left and right lanes that it finds are similar enough and roughly parallel, which it does by comparing the standard deviation from of the two computed lane curvatures to the mean of the two computed curvatures. The code for this is in lines 452 to line 460 in `lane_find.py`.

![alt text][image6]

During this stage, a second degree polynomial line is fitted to each set of lane pixels using the numpy `polyfit()` method. If the image is a frame from a video, the previous frame's lines are averaged with 1/3 weight to smoothen transitions.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature in lines 394 through 413 in my code in `lane_find.py`. It works by approximating conversion from pixel distance to meters, then applying [the formula for measuring the radius of curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) to the bottom of the image.

My code for measuring lane center offset is located in lines 418 though 435 `lane_find.py`. It works by approximately converting from pixel distance to meters, measuring the distance from the left lane to the left image edge and from the right lane to the right image edge, then calculating how much offset would be required to make them equal.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

My complete image processing pipeline is in lines 465 through 527 in my code in `lane_find.py` in the function `image_pipeline()`. Here's an example from the completed pipeline:

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project felt like a whirlwind of tricky ways to do very specific things. I can tell that that a lot of time and thinking has gone into developing these techniques, and some of them really blew my mind! I am very glad to start my introduction to computer vision with such specific goals as drawing lane lines, because I'm getting a sense for how much sidetracking I would be doing if I didn't have such a focused project.

That said, this project was pretty challenging to me because I did still need to sidetrack, read, and think about why these techniques work. The project guidelines offered all sorts of awesome suggestions about ways to do this project well, but I feel like I didn't even get half of them fully implimented because I was still tinkering around with the basics. So while I had a great return on investment in terms of learnings for my time spent, my pipeline is still very limited in its application.

My pipeline fails any time there is excessive shadowing or glare, or if the curvature is too extreme. So it makes it through the project video, but still has tons of room for improvement. I found that I got better results setting the thresholds to be lenient, which make for smoother highlighting but intensifies the problems with shadowing and glares.

I think that my pipeline's primary problem right now is that I don't have my technique for isolating white and yellow colors quite right. I also suspect that something may be going strange with moviepy's images - I've noticed that if I save an image straight from a clip, the colors aren't quite right, despite it supposedly being in RGB format. I think I need some more experience with color channels before I'm going to be able to make a method that's quite right for isolating possible lane colors. Otherwise, my histogram line fit method could use some work - I'd like to do something that's a little more iterative in terms of check whether my result makes sense. But I think my work on this project was a step in learning about computer vision techniques!

