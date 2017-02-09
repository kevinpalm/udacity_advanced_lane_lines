import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


class LaneFinder:
    """After calibrating, finds and highlights the lane in images and videos"""

    def __init__(self, calibration_glob, nx, ny, visualization=False):

        # Init by calibrating the camera
        print("Calibrating camera...")

        # Create placeholder arrays for appending to
        objp = []
        imgp = []

        # Find the calibration pictures
        cals = glob.glob(calibration_glob)

        for fname in cals:

            # Read in each image
            img = cv2.imread(fname)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Try and find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If corners we found for this image...
            if ret is True:

                # Initialize the object points
                imobjp = np.zeros((nx*ny, 3), np.float32)
                imobjp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

                # Append each set of points to the corresponding set
                objp.append(imobjp)
                imgp.append(corners)

            else:
                print("Couldn't find corners for {}".format(fname))

        # Save the camera calibrations
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objp, imgp, gray.shape[::-1], None, None)

        # Define and save the top down transforms
        sourcepoints = np.float32([[int(img.shape[1]*7/12), int(img.shape[0]*7/11)],
                                   [int(img.shape[1]), int(img.shape[0])],
                                   [0, int(img.shape[0])],
                                   [int(img.shape[1]*5/12), int(img.shape[0]*7/11)]])
        destinationpoints = np.float32([[img.shape[1], 0],
                                        [img.shape[1], img.shape[0]],
                                        [0, img.shape[0]],
                                        [0, 0]])
        self.M = cv2.getPerspectiveTransform(sourcepoints, destinationpoints)
        self.Minv = cv2.getPerspectiveTransform(destinationpoints, sourcepoints)

        if visualization is True:

            # Transform the last calibration image for a dummy check / visualization
            cimg = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            fig = plt.figure(figsize=(10, 3))
            a = fig.add_subplot(1, 2, 1)
            imgplot = plt.imshow(img)
            a.set_title('Original')
            a = fig.add_subplot(1, 2, 2)
            imgplot = plt.imshow(cimg)
            a.set_title('Undistorted')
            plt.savefig(fname.replace("camera_cal/", "output_images/corrected_"))

    def color_threshold(self, image, visualization=False):

        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        # TODO: Do some exploring for better color channels...
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = 170
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        if visualization is True:
        # Plotting thresholded images
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.set_title('Stacked thresholds')
            ax1.imshow(color_binary*255)

            ax2.set_title('Combined S channel and gradient thresholds')
            ax2.imshow(combined_binary, cmap='gray')
            plt.savefig("output_images/color_threshold.jpg")

        return combined_binary

    def top_down_perspective_transform(self, image, visualization=False):

        # Undistort the image
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        # Transform the input image to a top-down view
        warped = cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        return warped

    def inverse_top_down_perspective_transform(self, image, visualization=False):

        # Undistort the image
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        # Transform the input image to a top-down view
        warped = cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        return warped

    def histogram_fit(self, image, visualisation=False):

        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[int(image.shape[0] / 2):, :], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image)) * 255

        # Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9

        # Set height of windows
        window_height = np.int(image.shape[0] / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if visualisation is True:

            # Generate x and y values for plotting
            ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig("output_images/histogram_fit.jpg")

        return left_fit, leftx, lefty, right_fit, rightx, righty

    def margin_fit(self, image, pastleft, pastright, visualization=False):

        # Extract active pixels
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define margin of search
        margin = 100

        # Apply margin along previous lines
        left_lane_inds = (
        (nonzerox > (pastleft[0] * (nonzeroy ** 2) + pastleft[1] * nonzeroy + pastleft[2] - margin)) & (
        nonzerox < (pastleft[0] * (nonzeroy ** 2) + pastleft[1] * nonzeroy + pastleft[2] + margin)))
        right_lane_inds = (
        (nonzerox > (pastright[0] * (nonzeroy ** 2) + pastright[1] * nonzeroy + pastright[2] - margin)) & (
        nonzerox < (pastright[0] * (nonzeroy ** 2) + pastright[1] * nonzeroy + pastright[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        if visualization is True:

            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((image, image, image)) * 255
            window_img = np.zeros_like(out_img)

            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig("output_images/margin_fit.jpg")

        return left_fit, leftx, lefty, right_fit, rightx, righty

    def format_lines(self, image, left_fit, right_fit):

        # Fit the line coeficients to lines
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return ploty, left_fitx, right_fitx

    def draw_lines(self, warped, image, ploty, left_fitx, right_fitx):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.inverse_top_down_perspective_transform(color_warp)

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        return result

    def image_pipeline(self, image):

        # Apply color channel and sorbel
        img = self.color_threshold(image)

        # Transform perspective
        img = self.top_down_perspective_transform(img)

        # Use the histogram method to draw lane lines
        left_fit, leftx, lefty, right_fit, rightx, righty = self.histogram_fit(img)

        # Format the lines
        ploty, left_fitx, right_fitx = self.format_lines(img, left_fit, right_fit)

        # Draw lane on the original image
        image = self.draw_lines(img, image, ploty, left_fitx, right_fitx)

        return image

def main():

    lf = LaneFinder("camera_cal/calibration*.jpg", 9, 6)
    image = cv2.imread("test_images/test1.jpg")
    image = lf.image_pipeline(image)
    cv2.imwrite("test.jpg", image)

if __name__ == '__main__':
    main()