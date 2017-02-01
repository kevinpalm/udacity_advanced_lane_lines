import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


class LaneFinder:

    def __init__(self, calibration_glob, nx, ny):

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

                # Save a copy of the image with points drawn on it
                drawimg = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                cv2.imwrite(fname.replace("camera_cal", "camera_cal_points_drawn"), drawimg)

                # Append each set of points to the corresponding set
                objp.append(imobjp)
                imgp.append(corners)

            else:
                print("Couldn't find corners for {}".format(fname))

        # Calibrate the camera
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objp, imgp, gray.shape[::-1], None, None)

        # Transform the calibration images for a dummy check
        for fname in cals:
            img = cv2.imread(fname)
            img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            cv2.imwrite(fname.replace("camera_cal", "camera_cal_corrected"), img)

def main():
    lf = LaneFinder("camera_cal/calibration*.jpg", 9, 6)

if __name__ == '__main__':
    main()