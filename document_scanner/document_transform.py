import numpy as np
import cv2

"""
Implementation of some helper functions for our document scanner

- Point ordering gives a consistent ordering of the four corners of a rectangle
- Document transform is the main functionality that we need, transforming the image so we get a nice view of the document    
"""

def point_ordering(points):
    rectangle = np.zeros((4, 2), dtype = "float32")
    # Sum the x and y coordinates of each point
    summed = points.sum(axis = 1)
    # Highest sum is the top-left point and the smallest sum is the bottom-right one
    rectangle[0] = points[np.argmin(summed)]
    rectangle[2] = points[np.argmax(summed)]
    
	# Determine the difference in coordinates of each point
    difference = np.diff(points, axis = 1)
    # The largest difference will be the bottom-left, while the smallest will be top-right
    rectangle[1] = points[np.argmin(difference)]
    rectangle[3] = points[np.argmax(difference)]
	# return the ordered coordinates
    return rectangle


def document_transform(image, points):
	# Make a consistent ordering of the points
	rectangle = point_ordering(points)
	(tl, tr, br, bl) = rectangle
 
	# New width will be the largest width of the rectangle
	bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	new_width = max(int(bottom_width), int(top_width))
 
	# Do the same fo the height
	right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	new_height = max(int(right_height), int(left_height))
 
	# Use new width and height to determine the destination points of the rectangle
	destination = np.array([
		[0, 0],
		[new_width - 1, 0],
		[new_width - 1, new_height - 1],
		[0, new_height - 1]], dtype = "float32")
 
	# Use opencv to determine the transformation matrix and then warp the image with this matrix
	T = cv2.getPerspectiveTransform(rectangle, destination)
	warped = cv2.warpPerspective(image, T, (new_width, new_height))
	return warped