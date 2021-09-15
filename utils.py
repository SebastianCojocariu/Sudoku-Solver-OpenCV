import cv2
import numpy as np
import os
import onnx
from onnx_tf.backend import prepare
from matplotlib import pyplot as plt
from collections import defaultdict
import utils
import scipy
import math
import itertools
import argparse
import copy

from skimage.draw import line
from scipy import special
from scipy import ndimage

# Converts a RGB photo to Black and White + some repair techniques (morphological)
def transform_to_black_and_white(image, ksize=55, C=21, k_blur=11, k_cross=9):
	image = image.copy()
	# Smooth the image
	image = cv2.GaussianBlur(image, (k_blur, k_blur), 0)
	# convert from RGB to gray
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# convert to black and white image
	image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, ksize, C)
	# Cover holes
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_cross, k_cross))
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
	
	# return the black and white image
	return image


# Takes a black and white image, find the largest contour using floodfill (from get_largest_component)
# Then, it detects the sudoku contour based on the outer_grid determined.  Next, we will remove the outer
# grid and we will map the countour twice: one time for the numbers inside the grid, and one time for the
# grid lines (while applying a perspective transformation: see transform_perspective function)
def split_lines_from_numbers(image_black_and_white, width=2000, height=2000):
	sudoku_outer_grid_image = get_largest_component(image=image_black_and_white)
	sudoku_contour_points = detect_sudoku_contour(image=sudoku_outer_grid_image)

	sudoku_numbers_only = remove_image_from_image(remove_image=sudoku_outer_grid_image, from_image=image_black_and_white)

	sudoku_numbers_image = transform_perspective(image=sudoku_numbers_only, contour=sudoku_contour_points, width=width, heigth=height)
	sudoku_grid_image = transform_perspective(image=sudoku_outer_grid_image, contour=sudoku_contour_points, width=width, heigth=height)

	return sudoku_numbers_image, sudoku_grid_image


# Detects the largest area based on findContours function. It only choose the largest area that can be approximated
# by a quadrilateral (by varying the contour error through epsilon)
def detect_sudoku_contour(image):
	assert len(image.shape) == 2, "image should be black & white or gray"
	
	image = image.copy()
	area_image = np.prod(image.shape)
	width, height = image.shape

	contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	largest_contour, max_area_so_far = None, float("-Infinity")

	for contour in contours:
		area_curr_contour = cv2.contourArea(contour)
		if max_area_so_far < area_curr_contour and area_curr_contour < area_image * 0.97:
			perimeter = cv2.arcLength(contour, True)
			minimum_difference = float("Infinity")
			for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.2]:
				approximation_polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
				approximation_area = cv2.contourArea(approximation_polygon)
				if len(approximation_polygon) == 4 and abs(approximation_area - area_curr_contour) <= minimum_difference:
					largest_contour, max_area_so_far = approximation_polygon, area_curr_contour
					minimum_difference = abs(approximation_area - area_curr_contour)
	
	# if there exists a largest contour with 4 points
	if largest_contour is not None:
		# order the points in this order: 0 -> 1
		#								  3 <- 2	  	
		
		largest_contour = np.squeeze(largest_contour)

		largest_contour = [largest_contour[i] for i in range(len(largest_contour))]
		largest_contour.sort(key=lambda x: (x[0], x[1]))
		
		if largest_contour[0][1] - largest_contour[1][1] > height * 0.1:
			largest_contour[0], largest_contour[1] = largest_contour[1], largest_contour[0]

		if largest_contour[2][1] - largest_contour[3][1] > height * 0.1:
			largest_contour[2], largest_contour[3] = largest_contour[3], largest_contour[2]

		largest_contour = np.expand_dims(np.asarray([largest_contour[0], largest_contour[2], largest_contour[3], largest_contour[1]]), axis=0)
	else:
		raise Exception("Could not find a rectangle containing the sudoku")

	# return the largest contour
	return largest_contour


# Returns the largest component from image through floodfilling
def get_largest_component(image):
	image = image.copy()
	
	# repair the connections between lines 
	image = cv2.dilate(image, np.ones((7, 7)), iterations=1)

	# floodfilling image
	flood_fill_image = np.float32(image.copy())

	# start floodfilling for each component and update the maximum component found so far
	# for this part we use a value different from 0 and 255 (64 was chosen for simplicity)
	max_area, contour_sudoku = 0, None 
	for i in range(flood_fill_image.shape[0]):
		for j in range(flood_fill_image.shape[1]):
			if flood_fill_image[i][j] >= 200:
				(area, _, _, _) = cv2.floodFill(flood_fill_image, None, seedPoint=(j, i), newVal=64)
				if area > max_area:
					seedPoint = (j, i)
					max_area = area

	# floodfill again the largest contour with white
	cv2.floodFill(flood_fill_image, None, seedPoint=seedPoint, newVal=255)
	
	# make all the remaining components black through floodfilling.
	for i in range(flood_fill_image.shape[0]):
		for j in range(flood_fill_image.shape[1]):
			if flood_fill_image[i][j] == 64 and (j, i) != seedPoint:
				flood_fill_image[i][j] = 0

	# repair the lines again
	flood_fill_image = cv2.dilate(flood_fill_image, np.ones((5, 5)), iterations=1)
	
	return np.uint8(flood_fill_image)


# Finds a perspective transformation from a specific contour from an image to a rectangle of specified width and height
def transform_perspective(image, contour, width, heigth):
	assert contour is not None, "contour passed is None"
	image, contour = image.copy(), contour.copy()

	contour = np.float32(contour)
	contour = np.squeeze(contour)
	
	image = np.float32(image)

	# compute the 4 points that determine the rectangle
	destination_info = np.array([[0, 0], [width - 1, 0], [width - 1, heigth - 1], [0, heigth - 1]], dtype = "float32")

	# find the transformation matrix
	M, _ = cv2.findHomography(contour, destination_info, cv2.RANSAC, 5.0) # it was method=0 before

	image = cv2.warpPerspective(image, M, (heigth, width))
	
	# back to black and white image (if it's not already)
	image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

	# repair holes
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
	
	# return the image
	return np.uint8(image)


# Given an image, it find the largest contour, find the corresponding enclosing rectangle
# and center the contour. A new image is returned 
def center_digit(image):
	image = image.copy()
	width, height = image.shape
	area_image = np.prod(image.shape)

	# find largest contour
	contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	largest_contour, max_area_so_far = None, float("-Infinity")

	for contour in contours:
		area_curr_contour = cv2.contourArea(contour)
		if max_area_so_far < area_curr_contour and area_curr_contour < area_image * 0.97:
			largest_contour = contour
			max_area_so_far = area_curr_contour

	# Get the bounding rectangle and calculate the new coordinates for the centered left-top corner
	if largest_contour is not None:
		x, y, w, h = cv2.boundingRect(largest_contour)
		new_x = width // 2 - w // 2
		new_y = height // 2 - h // 2

		# apply a translation
		dx = new_x - x
		dy = new_y - y
		M = np.float32([[1, 0, dx], [0, 1, dy]])
		image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# return the image
	return image

# Helper that shows multiple images in a grid format (specified by nrows and ncols)
def show_images(images_list, nrows=2, ncols=3):
	for i, image in enumerate(images_list):
		plt.subplot(nrows, ncols, i + 1), 
		plt.imshow(image, 'gray', vmin=0, vmax=255)
	plt.show()

# Removes an image from another image
def remove_image_from_image(remove_image, from_image):
	new_image = from_image.copy()
	for i in range(new_image.shape[0]):
		for j in range(new_image.shape[1]):
			if remove_image[i][j] >= 200:
				new_image[i][j] = 0
	return new_image
		
# Finds the number of pixels in a given image. The percent is used
# to evenly shrink the region of interest
def find_no_white_pixels_from_image(image, percent=0.0):
	no_white_pixels = 0
	
	width, height = image.shape
	width_left, width_right = int(percent * width), int(width * (1 - percent))
	heigth_left, height_right = int(percent * height), int(height * (1 - percent))
	
	for i in range(width_left, width_right):
		for j in range(heigth_left, height_right):
			if image[i][j] >= 200:
				no_white_pixels += 1
	
	return no_white_pixels

# Splits an image into bounding boxes. Each bounding box has the structure: (center, width, height)
def split_into_bboxes(image, no_cells=9):
	(width_image, heigth_image) = image.shape
	width_step, heigth_step = width_image // no_cells, heigth_image // no_cells

	res = []
	for i in range(no_cells):
		aux = []
		for j in range(no_cells):
			mid_x = min(width_image - 1, int((i + 1/2) * width_step))
			mid_y = min(heigth_image - 1, int((j + 1/2) * heigth_step))
			aux.append([(mid_x, mid_y), width_step, heigth_step])
		res.append(aux)
	
	return res


# Converts a boundingbox specified by (center, width, height) into the corresponding
# (row_left: row_right), (col_left: col_right) (checking for boundary issues that might occur)
def convert_bbox(image, bbox):
	mid, width, height = bbox
	row_left, row_right = max(0, mid[0] - width//2), min(image.shape[0], mid[0] + width//2)
	col_left, col_right = max(0, mid[1] - height//2), min(image.shape[1], mid[1] + height//2)
	
	return (row_left, row_right), (col_left, col_right)

# Extracts the numbers from an image with numbers (arranged as per sudoku description)
def extract_numbers_from_image(sudoku_image, bboxes, verbose=0):
	assert len(sudoku_image.shape) == 2, "sudoku_image must be black/white or gray, not RGB"

	no_cells = len(bboxes)

	# for each bbox check the number of white pixels and store them. Multiples images must be black
	# as a lot of cell dont have any numbers in it
	white_pixels_values = []
	stored_values = [[0] * no_cells for _ in range(no_cells)]
	for i in range(no_cells):
		for j in range(no_cells):
			(row_left, row_right), (col_left, col_right) = convert_bbox(image=sudoku_image, bbox=bboxes[i][j])
			image = sudoku_image[row_left: row_right, col_left: col_right]
			
			white_pixels_count = find_no_white_pixels_from_image(image=image, percent=0.1)
			
			white_pixels_values.append(white_pixels_count)
			stored_values[i][j] = white_pixels_count

	# Calculate a suitable threshold for the number of white pixels
	mean = np.mean(white_pixels_values)
	std = np.std(white_pixels_values)

	threshold = mean
	if verbose:
		print("Threshold: ", threshold)
	
	# filter based on this threshold
	matrix = [["o"] * no_cells for _ in range(no_cells)]
	for i in range(no_cells):
		for j in range(no_cells):
			if stored_values[i][j] >= threshold:
				matrix[i][j] = "x"

	return matrix

	
# Some helper function that might be of use for a better interpolation scheme.
# One could find the connected components and try to interpolate the celss until 
# the entire sudoku is filled.

'''
def are_neighbours(rectangle1, rectangle2):
	mid1, w1, h1 = rectangle1
	mid2, w2, h2 = rectangle2

	m_w, m_h = (w1 + w2) // 2, (h1 + h2) // 2

	if abs(mid1[0] - mid2[0]) <= m_w // 2 and abs(abs(mid1[1] - mid2[1]) -  m_h) <= m_h / 2:
		return True

	if abs(mid1[1] - mid2[1]) <= m_h // 2 and abs(abs(mid1[0] - mid2[0]) -  m_w) <= m_w / 2: 
		return True

	return False


def are_on_the_same_line(rectangle1, rectangle2):
	mid1, w1, h1 = rectangle1
	mid2, w2, h2 = rectangle2

	m_w, m_h = (w1 + w2) // 2, (h1 + h2) // 2

	if abs(mid1[0] - mid2[0]) <= m_w // 2:
		return True

	if abs(mid1[1] - mid2[1]) <= m_h // 2:
		return True

	return False


def is_inside_point(point, rectangle):
	mid1, w1, h1 = rectangle

	if not (point[0] >= mid[0] - w1 // 2 and point[0] <= mid[0] + w1 // 2):
		return False
	if not (point[1] >= mid[1] - h1 // 2 and point[1] <= mid[0] + h1 // 2):
		return False

	return True


def interpolate(connected_components, rectangle1, rectangle2):
	mid1, w1, h1 = rectangle1
	mid2, w2, h2 = rectangle2

	m_w, m_h = (w1 + w2) // 2, (h1 + h2) // 2

	dx, dy = mid1[0] - mid2[0], mid1[1] - mid2[1]
	
	no_rectangles = min(math.round(abs(dx - m_w) / m_w), math.round( abs(dy - m_h) / m_h))

	if no_rectangles == 0:
		return []
	
	dx_sign = 0 if abs(dx) <= 0.001 else abs(dx) / dx
	dy_sign = 0 if abs(dy) <= 0.001 else abs(dy) / dy
	
	m_w = abs(dx - m_w) / n
	m_h = abs(dy - m_h) / n

	# TO DO
	res = []
	for i in range(1, n):
		x_coord = mid1[0] + i / (n + 1) * (mid2[0] - mid1[0])
		y_coord = mid1[1] + i / (n + 1) * (mid2[1] - mid1[1])
		current_rectangle = ((x_coord, y_coord), m_w, m_h)
		res.append(current_rectangle)

		# if any rectangle from connected components has their center inside the new interpolated rectange => cannot interpolate
		for (center, _, _) in connected_components:
			if is_inside_point(point=center, rectangle=current_rectangle):
				return []

	return res
'''