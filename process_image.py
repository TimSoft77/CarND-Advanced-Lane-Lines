import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


# Calculate radius of curvature from a parabola
def radius_curvature(a, b, y):
    return ((1+(2*a*y+b)**2)**1.5)/(abs(2*a))


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # distance in pixels of vehicle center from left of image
        self.line_base_pos = None
        # number of consecutive iterations for which no good lane was found
        self.num_errors = 0

    # max last succuessful fits to average together to get the new line
    n_recent = 10

    # Calculate an average line of recent, successful line searches, giving greater weight to more recent lines
    def wt_avg_recent_xfitted(self):
        num_recent = len(self.recent_xfitted)
        x_fit = np.zeros_like(self.recent_xfitted[0])  # numerator
        den = 0  # denominator
        for ii in range(num_recent):
            x_fit += self.recent_xfitted[ii]*(ii+1)/num_recent
            den += (ii+1)/num_recent
        return x_fit/den


# Initialize left and right lane lines
left_lane_line = Line()
right_lane_line = Line()


def process_image(img, plot_graphs=False):
    # This expects images in RGB format
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Undistort
    mtx = pickle.load(open('cal_mtx.p', 'rb'))
    dist = pickle.load(open('cal_dist.p', 'rb'))
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

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
    s_thresh_min = 120
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Perspective transform
    offset = 300
    src = np.float32([[591, 453], [687, 453], [1002, 653], [302, 653]])
    dst = np.float32([[offset, 0], [1280 - offset, 0], [1280 - offset, 720], [offset, 720]])
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)  # Needed much later
    binary_warped = cv2.warpPerspective(combined_binary, m, combined_binary.shape[::-1])

    # Find lane lines
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int(histogram.shape[0] / 2)
    # Set the maximum frames back that the base positions will be taken before they are reset
    max_errors_before_reset = 10
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines, unless previous starting points have been found
    if left_lane_line.line_base_pos is not None and left_lane_line.num_errors < max_errors_before_reset:
        leftx_base = left_lane_line.line_base_pos
    else:
        leftx_base = np.argmax(histogram[:midpoint])
    if right_lane_line.line_base_pos is not None and right_lane_line.num_errors < max_errors_before_reset:
        rightx_base = right_lane_line.line_base_pos
    else:
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 200
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their median position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if window == 0:
                left_lane_line.line_base_pos = leftx_current
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            if window == 0:
                right_lane_line.line_base_pos = rightx_current

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each (exception handling in case no points are found - needed for challenge video)
    left_fit = [2, 1, 1]
    right_fit = [2, 1, 1]
    fit_worked = True
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except:
        fit_worked = False
    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        fit_worked = False

    # Generate x and y values for evaluating lines
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    diff_fitx = right_fitx - left_fitx
    range_diff_fitx = np.max(diff_fitx) - np.min(diff_fitx)

    # Decide which, if either, lane lines are acceptable
    max_ok_range_diff_fitx = 150  # pixels
    if range_diff_fitx <= max_ok_range_diff_fitx and fit_worked:
        left_lane_line.num_errors = 0
        left_lane_line.recent_xfitted.append(left_fitx)
        if len(left_lane_line.recent_xfitted) > left_lane_line.n_recent:
            left_lane_line.recent_xfitted.pop(0)
        right_lane_line.num_errors = 0
        right_lane_line.recent_xfitted.append(right_fitx)
        if len(right_lane_line.recent_xfitted) > right_lane_line.n_recent:
            right_lane_line.recent_xfitted.pop(0)
    else:
        left_lane_line.num_errors += 1
        right_lane_line.num_errors += 1

    # Get the smoothed lines, and plot them
    if len(left_lane_line.recent_xfitted) > 0:
        left_fitx = left_lane_line.wt_avg_recent_xfitted()
    if len(right_lane_line.recent_xfitted) > 0:
        right_fitx = right_lane_line.wt_avg_recent_xfitted()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(out_img, np.int_(pts_left), False, (255, 255, 0), thickness=3)
    cv2.polylines(out_img, np.int_(pts_right), False, (255, 255, 0), thickness=3)

    # Draw the lane onto the warped blank image
    color_warp = np.zeros_like(out_img)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Draw the lane in the original perspective
    lane_only = cv2.warpPerspective(color_warp, m_inv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB), 1, lane_only, 0.3, 0)

    # Print text on image
    # Calculate the position of the center of the car
    center_car = (leftx_base + rightx_base)/2
    # Calculate the radii of curvature
    left_rad_curve = radius_curvature(left_fit[0], left_fit[1], binary_warped.shape[0])
    right_rad_curve = radius_curvature(right_fit[0], right_fit[1], binary_warped.shape[0])
    text = ['Rad curves=' + str(round(left_rad_curve, 1)) + ', ' + str(round(right_rad_curve, 1)), 'Midpt=' + str(round(center_car, 0)), 'num_error=' + str(left_lane_line.num_errors)]
    y_text_pos = 0
    for t in text:
        retval, baseline = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
        cv2.putText(result, t, (binary_warped.shape[1] - retval[0], retval[1] + baseline + y_text_pos), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=4)
        y_text_pos += retval[1] + baseline

    # Plotting images
    if plot_graphs:
        plt.figure(1, figsize=(20, 10))
        plt.subplot(331)
        plt.title('Original')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.subplot(332)
        plt.title('Undistorted')
        plt.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
        plt.subplot(333)
        plt.title('Stacked thresholds')
        plt.imshow(color_binary)
        plt.subplot(334)
        plt.title('Combined S channel and gradient thresholds')
        plt.imshow(combined_binary, cmap='gray')
        plt.subplot(335)
        plt.title('Combined binary warped')
        plt.imshow(binary_warped, cmap='gray')
        plt.subplot(336)
        plt.title('Lane lines drawn warped')
        plt.imshow(out_img)
        plt.subplot(337)
        plt.title('Lane only, warped')
        plt.imshow(color_warp)
        plt.subplot(338)
        plt.title('Lane only, camera perspective')
        plt.imshow(lane_only)
        plt.subplot(339)
        plt.title('Final result')
        plt.imshow(result)
        plt.show()

    return result

# Read the image in RGB format - for test purposes
image = cv2.cvtColor(cv2.imread('./test_images/test4.jpg'), cv2.COLOR_BGR2RGB)
# process_image(image, plot_graphs=True)

# Create video
output_name = 'project_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
output_clip.write_videofile(output_name, audio=False)
