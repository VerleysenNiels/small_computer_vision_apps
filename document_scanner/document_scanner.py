import cv2
import logging
import numpy as np

IMAGE_PATH = "images/example.jpg"

if __name__ == "__main__":
    # Load in the image and keep a copy
    image = cv2.imread(IMAGE_PATH)
    orig = image.copy()
    
    # Resize
    ratio = image.shape[0] / 750.0
    width = int(image.shape[1] / ratio)
    height = int(image.shape[0] / ratio)
    image = cv2.resize(image, (width, height))
    
    # Turn into grayscale and apply the canny edge detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    # Find contours in the detected edges
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    
    # Now let's check the contours to see if we can find the document
    for contour in contours:
        # Make an approximation of this contour
        peri = cv2.arcLength(contour, True)
        approximate_contour = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # As a simplified rule of thumb we can say that if the contour has four corners, we have found the document 
        if len(approximate_contour) == 4:
            document_contour = approximate_contour
            break
    
    # Show the different steps of the process
    cv2.imshow("Image", image)
    cv2.imshow("Edges", edged)
    cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()