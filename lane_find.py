import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

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
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), True)

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
        sourcepoints = np.float32([[int(img.shape[1]*217/384), int(img.shape[0]*28/44)],
                                   [int(img.shape[1]), int(img.shape[0])],
                                   [0, int(img.shape[0])],
                                   [int(img.shape[1]*167/384), int(img.shape[0]*28/44)]])
        destinationpoints = np.float32([[img.shape[1], 0],
                                        [img.shape[1], img.shape[0]],
                                        [0, img.shape[0]],
                                        [0, 0]])
        self.M = cv2.getPerspectiveTransform(sourcepoints, destinationpoints)
        self.Minv = cv2.getPerspectiveTransform(destinationpoints, sourcepoints)

        # Create a placeholder for past line fits
        self.past_fits = None

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
            plt.clf()


    def color_threshold(self, image, visualization=False):

        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        # Convert to HSV to separate out the desired colors
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Prep yellow mask
        yellow_min = np.array([20, 0, 0])
        yellow_max = np.array([60, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

        # Prep white mask
        white_min = np.array([0, 0, 220])
        white_max = np.array([255, 80, 255])
        white_mask = cv2.inRange(hsv, white_min, white_max)

        # Combine the masks
        color_mask = np.array((white_mask+yellow_mask)>0)

        # Prepare a grayscale image and mask it
        masked_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) * color_mask

        # Sobel x
        sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        bottom_half_scaled_sobel = scaled_sobel[int(scaled_sobel.shape[0] / 2):, :]
        thresh_min = np.mean(bottom_half_scaled_sobel)+np.std(bottom_half_scaled_sobel)*3.4
        thresh_max = 255
        # print("Sorbel threshold: min - ", thresh_min, "max - ", thresh_max)
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        bottom_half_s_channel = s_channel[int(s_channel.shape[0] / 2):, :]
        s_thresh_min = np.mean(bottom_half_s_channel)+np.std(bottom_half_s_channel)*3.4
        s_thresh_max = 255
        # print("S Channel threshold: min - ", s_thresh_min, "max - ", s_thresh_max)
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
            plt.clf()

        return combined_binary


    def undistort(self, image, visualization=False):

        # Undistort the image
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        if visualization is True:
            cv2.imwrite("output_images/undistorted.jpg", image)

        return image


    def top_down_perspective_transform(self, image):

        # Transform the input image to a top-down view
        warped = cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        return warped


    def inverse_top_down_perspective_transform(self, image, visualization=False):

        # Transform the input image to a top-down view
        warped = cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

        return warped


    def histogram_fit(self, image, visualisation=True):

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
            plt.clf()

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
            plt.clf()

        return left_fit, leftx, lefty, right_fit, rightx, righty


    def format_lines(self, image, left_fit, right_fit):

        # Fit the line coeficients to lines
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        return ploty, left_fitx, right_fitx


    def draw_lane(self, warped, image, ploty, left_fitx, right_fitx, leftx, lefty, rightx, righty):

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp2 = color_warp.copy()

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Add some red to the left lane pixels
        for x, y in zip(leftx, lefty):
            color_warp2[y, x, 0] = 255

        # Add some blue to the right lane pixels
        for x, y in zip(rightx, righty):
            color_warp2[y, x, 2] = 255

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.inverse_top_down_perspective_transform(color_warp)
        newwarp2 = self.inverse_top_down_perspective_transform(color_warp2)

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        result = cv2.addWeighted(result, 1, newwarp2, 1, 0)

        return result


    def find_curvature(self, ploty, leftx, lefty, rightx, righty):

        # Define where we we evaluate curvature - the bottom of the image
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(np.multiply(lefty, ym_per_pix), np.multiply(leftx, xm_per_pix), 2)
        right_fit_cr = np.polyfit(np.multiply(righty, ym_per_pix), np.multiply(rightx, xm_per_pix), 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # Now our radius of curvature is in meters
        return left_curverad, right_curverad


    def find_offset(self, left_fit, right_fit, image_shape):

        # Save the image dimensions
        height = image_shape[0]
        width = image_shape[1]

        # Define conversions in x and y from pixels space to meters
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Find where the lines interset with the base of the image
        left_fitx = left_fit[0] * height ** 2 + left_fit[1] * height + left_fit[2]
        right_fitx = right_fit[0] * height ** 2 + right_fit[1] * height + right_fit[2]

        # Calculate offset in pixels
        left_wing = width-right_fitx
        right_wing = width-abs(width-left_fitx)
        shift = min(left_wing, right_wing)
        offset = round(((left_wing-shift)+(right_wing-shift))/2.0*xm_per_pix, 2)

        return offset


    def draw_text(self, image, curvature, center):

        # Draw text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Radius of curvature (m): {}'.format(curvature),
                    (10, 30), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
        cv2.putText(image, 'Offset from the lane center (m): {}'.format(center),
                    (10, 60), font, 1, (200, 255, 155), 2, cv2.LINE_AA)

        return image


    def check_lines(self, left_curverad, right_curverad):

        # Get stats
        stdev = np.std(np.array([left_curverad, right_curverad]))
        mean = np.mean(np.array([left_curverad, right_curverad]))

        # Check values
        if (stdev/mean <= 0.8) and (stdev/mean != 0.0):
            return True
        else:
            return False


    def image_pipeline(self, image, past_fits=None):

        # Correct for lense distortion
        image = self.undistort(image)

        # Apply color channel and sorbel
        img = self.color_threshold(image)

        # Transform perspective
        img = self.top_down_perspective_transform(img)

        # If no prior lines are supplied, just use the histogram method
        if past_fits is None:

            # Use the histogram method to draw lane lines
            left_fit, leftx, lefty, right_fit, rightx, righty = self.histogram_fit(img)

            # Format the lines for drawing
            ploty, left_fitx, right_fitx = self.format_lines(img, left_fit, right_fit)

        else:

            try:

                # Try and use the margin fit method instead
                left_fit, leftx, lefty, right_fit, rightx, righty = self.margin_fit(img, past_fits[0], past_fits[1])

                # Apply an average against the last lines to smoothen
                left_fit = (np.array(left_fit)*2 + np.array(past_fits[0])) / 3
                right_fit = (np.array(right_fit)*2 + np.array(past_fits[1])) / 3

                # Format the lines points for drawing
                ploty, left_fitx, right_fitx = self.format_lines(img, left_fit, right_fit)

                # Find the curvature
                left_curverad, right_curverad = self.find_curvature(ploty, leftx, lefty, rightx, righty)

                check = self.check_lines(left_curverad, right_curverad)
                if check is False:
                    raise Exception("The margin fit method didn't pass the check.")

            except:

                # If margin fit doesn't work, go back to histogram method
                left_fit, leftx, lefty, right_fit, rightx, righty = self.histogram_fit(img)

                # Apply an average against the last lines to smoothen
                left_fit = (np.array(left_fit)*2 + np.array(past_fits[0])) / 3
                right_fit = (np.array(right_fit)*2 + np.array(past_fits[1])) / 3

                # Format the lines for drawing
                ploty, left_fitx, right_fitx = self.format_lines(img, left_fit, right_fit)


        # Draw lane on the original image
        image = self.draw_lane(img, image, ploty, left_fitx, right_fitx, leftx, lefty, rightx, righty)

        # Find the curvature
        left_curverad, right_curverad = self.find_curvature(ploty, leftx, lefty, rightx, righty)

        # Find the offset
        offset = self.find_offset(left_fit, right_fit, image.shape)

        # Draw the text on the image
        image = self.draw_text(image, int((left_curverad + right_curverad)/2.0), offset)

        return image, left_fit, right_fit


    def process(self, image):

        # Run the pipeline once
        image, left_fit, right_fit = self.image_pipeline(image, past_fits=self.past_fits)

        # Update the most recent past fits
        self.past_fits = [left_fit, right_fit]

        return image


    def process_image(self, fileloc):

        # Ensure the past fits are clear
        self.past_fits = None

        # Read the image
        image = cv2.imread(fileloc)

        # Run the pipeline
        image = self.process(image)

        # Clear the past fits for good housekeeping
        self.past_fits = None

        return image


    def process_clip(self, fileloc):

        # Ensure the past fits are clear
        self.past_fits = None

        # Load and define the clip
        clip = VideoFileClip(fileloc)

        # Apply the transform
        clip = clip.fl_image(self.process)

        # Clear the past fits for good housekeeping
        self.past_fits = None

        return clip


def main():

    # Initialize the lane finder
    lf = LaneFinder("camera_cal/calibration*.jpg", 9, 6)

    # Highlight all the test images
    for image in ["straight_lines1.jpg", "straight_lines2.jpg", "test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg",
                  "test5.jpg", "test6.jpg"]:
        processed_image = lf.process_image("test_images/"+image)
        cv2.imwrite("test_images/processed"+image, processed_image)

    # Highlight the video clips
    for cliploc in ["project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]:
        clip = lf.process_clip(cliploc)
        clip.write_videofile("processed_" + cliploc, audio=False)

if __name__ == '__main__':
    main()