import cv2
import logging
from collections import deque
from math import sqrt

# COLORS IN HSV SPACE
MIN_COLOR = (0, 100, 170)
MAX_COLOR = (15, 255, 255)
TRAIL_LENGTH = 10

if __name__ == "__main__":
     # Logging config
    logging.basicConfig(level=logging.INFO)
    
    # Set up a video stream
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    
    ret, frame = webcam.read()
    
    # Check if we could open the webcam
    if not webcam.isOpened():
        logging.error("Could not open the webcam")
        exit()
    
    history = deque(maxlen=TRAIL_LENGTH)
    
    while True:
        # Get the next frame
        ret, frame = webcam.read()
        
        # Check if we were able to capture a frame
        if not ret:
            logging.error("Unable to capture the next image")
            break 
        
        # Apply gaussian blurring
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        # Transform to HSV color space
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create a color mask
        masked = cv2.inRange(hsv_frame, MIN_COLOR, MAX_COLOR)
        # Remove small blobs (false detections)
        masked = cv2.erode(masked, None, iterations=2) 
        masked = cv2.dilate(masked, None, iterations=2)
        
        # Similar to the document scanner, find contours in the mask
        contours, _ = cv2.findContours(masked.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # We'll use this to remove old points when no detections are made anymore
        added_a_point = False
        
        # Let's find our tennisball
        center = None
        if len(contours) > 0:
            # Select the contour with the largest area
            selected_contour = max(contours, key=cv2.contourArea)
            
            # Determine a circle that encloses the selected contour
            ((x, y), radius) = cv2.minEnclosingCircle(selected_contour)
            
            # Determine the center of the ball
            moments = cv2.moments(selected_contour)
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            
            # If the radius is large enough we can draw the circle on the frame
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                # Add this point to the history
                history.appendleft(center)
                added_a_point = True
                
        if not added_a_point and len(history) > 0:
            history.pop()

        # Draw a trail with the previous points
        for i in range(1, len(history)):
            if history[i - 1] is None or history[i] is None:
                continue
            # Draw a line from point to point, where the thickness gets thinner as the points are further in the past
            thickness = int(sqrt(TRAIL_LENGTH / float(i + 1)) * 3)
            cv2.line(frame, history[i - 1], history[i], (255, 0, 0), thickness)
        
        # Display the frame
        cv2.imshow("frame", frame)
        
        # If the user presses "q" the script ends
        if cv2.waitKey(1) == ord("q"):
            break
               