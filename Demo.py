import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import ImageGrab
import cv2
import math
import time
import pyautogui
from directkeys import PressKey, ReleaseKey, W, A, S, D
#Original
def draw_lines(img, lines):
	try:
		for line in lines:
			coords = line[0]
			cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,0], 3)
	except:
		pass


def roi(img, vertices):
	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, 255)
	masked = cv2.bitwise_and(img, mask)
	return masked


def process_img(original_image):
	processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	processed_img = cv2.Canny(processed_img, threshold1 = 200, threshold2 = 300)
	processed_img = cv2.GaussianBlur(processed_img, (3,3), 0)
	vertices = np.array([[10,500], [10,300], [300,200], [500,200], [800, 300], [800, 500]], np.int32)
	processed_img = roi(processed_img, [vertices])

	lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 5)
	draw_lines(processed_img, lines)
	return processed_img


#Using improvement

def grayscale(img):
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def region_of_interest(img):
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img)
    
    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.95, rows]
    left_top = [cols * 0.4, rows * 0.6]
    right_top = [cols * 0.6, rows * 0.6]
    
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255, ) * mask.shape[2])
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	try:
		for line in lines:
			for x1,y1,x2,y2 in line:
				cv2.line(img, (x1, y1), (x2, y2), color, thickness)
	except:
		pass
	return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	draw_lines(line_img, lines)
	return line_img

def get_hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=300):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
	return cv2.addWeighted(initial_img, α, img, β, γ)

def to_hls(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
def to_hsv(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
def isolate_color_mask(img, low_thresh, high_thresh):
	assert(low_thresh.all() >=0  and low_thresh.all() <=255)
	assert(high_thresh.all() >=0 and high_thresh.all() <=255)
	return cv2.inRange(img, low_thresh, high_thresh)

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
	return cv2.LUT(image, table)
def process_img_improved(screen_grab):
	gray_screen = grayscale(screen_grab)
	darken_screen = adjust_gamma(gray_screen, 0.5)
	
	#using with RGB
	white_mask = isolate_color_mask(screen_grab, np.array([210, 210, 210], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
	yellow_mask = isolate_color_mask(screen_grab, np.array([100, 100, 0], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
	#Using HSL color space
	#white_mask = isolate_color_mask(to_hls(screen_grab), np.array([0, 210, 0], dtype=np.uint8), np.array([200, 255, 255], dtype=np.uint8))
	#yellow_mask = isolate_color_mask(to_hls(screen_grab), np.array([10, 0, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
	mask = cv2.bitwise_or(white_mask, yellow_mask)
	masked_screen = cv2.bitwise_and(darken_screen, darken_screen, mask = mask)
	blurred_screen = gaussian_blur(masked_screen, kernel_size = 7)
	
	canny_screen = canny(blurred_screen, 150, 300)


	vertices = np.array([[10,500], [10,450], [300,280], [500,280], [800, 450], [800, 500]], np.int32)
	processed_img = roi(canny_screen, [vertices])	
	cv2.imshow("masked", processed_img)
	lines = get_hough_lines(processed_img)
	processed_img = draw_lines(screen_grab, lines)

	return processed_img
	#return processed_img
#Testing
last_time = time.time()
def main():
	global last_time
	while(True):
		screen_grab = np.array(ImageGrab.grab(bbox = (0,40,800,600)))
		screen_grab = cv2.cvtColor(screen_grab, cv2.COLOR_BGR2RGB)
		#screen_grab = cv2.cvtColor(screen_grab, cv2.COLOR_BGR2RGB)
		#new_screen = lane_finder(screen_grab)
		#processed_screen = process_img(screen_grab)

		print("each frame took {} sceonds", format(time.time() - last_time))
		last_time = time.time()
		#cv2.imshow("window1", screen_grab)
		#cv2.imshow("window",cv2.cvtColor(screen_grab, cv2.COLOR_BGR2RGB))
		#cv2.imshow("window",processed_screen)
		#plt.imshow(screen_grab)
		#plt.show()

		processed_img = process_img_improved(screen_grab)

		cv2.imshow("oof", processed_img)

		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
main()