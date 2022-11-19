import argparse
import logging
import os

import cv2
from skimage.filters import threshold_local

from document_transform import document_transform

# Example image to be used as default
IMAGE_PATH = "images/example.jpg"

if __name__ == "__main__":
    # Logging config
    logging.basicConfig(level=logging.INFO)
    
    # Argument parsing
    parser = argparse.ArgumentParser(
        prog = "Document Scanner",
        description = "A script that takes a picture of a document and transforms it as if you would have scanned the document.")
    parser.add_argument("-i", "--image", default=IMAGE_PATH, dest="image_path", help="Image of the document to be scanned.")
    parser.add_argument("-r", "--recolor", default=False, dest="recolor", action=argparse.BooleanOptionalAction, help="Turn on or off recoloring of the scanned document.")
    parser.add_argument("-s", "--show_steps", default=False, dest="show", action=argparse.BooleanOptionalAction, help="Show the different steps taken by the scanner.")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        logging.error(f"Image {args.image_path} not found.")
        exit(-1)
    
    # Load in the image and keep a copy
    image = cv2.imread(args.image_path)
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
    
    if len(contours) == 0:
        logging.warning("No contours found in the image. Use the show_steps argument to get a better understanding of why your document is not being detected.")
        if args.show:
            cv2.imshow("Image", image)
            cv2.imshow("Edges", edged)      
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        exit(0)
    
    document_contour = None
    # Now let's check the contours to see if we can find the document
    for contour in contours:
        # Make an approximation of this contour
        peri = cv2.arcLength(contour, True)
        approximate_contour = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # As a simplified rule of thumb we can say that if the contour has four corners, we have found the document 
        if len(approximate_contour) == 4:
            document_contour = approximate_contour
            break
    
    if document_contour is None:
        logging.warning("Document not found in the image. Use the show_steps argument to get a better understanding of why your document is not being detected.")
        if args.show:
            cv2.imshow("Image", image)
            cv2.imshow("Edges", edged)
            cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 2)
            cv2.imshow("Outline", image)        
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        exit(0)
     
    warped = document_transform(orig, document_contour.reshape(4, 2) * ratio)
    ratio = warped.shape[0] / 750.0
    width = int(warped.shape[1] / ratio)
    height = int(warped.shape[0] / ratio)
    warped = cv2.resize(warped, (width, height))
    result = warped
    
    if args.recolor:
        # As a final step we can also recolor the warped image to have an even clearer scan
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresholds = threshold_local(warped, 11, offset = 10, method = "gaussian")
        recolored = (warped > thresholds).astype("uint8") * 255
        result = recolored
        
    if args.show:
        # Show the different steps of the process
        cv2.imshow("Image", image)
        cv2.imshow("Edges", edged)
        cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.imshow("Warped", warped)
        if args.recolor:
            cv2.imshow("Recolored", recolored)
    else:
        cv2.imshow("Document", result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
