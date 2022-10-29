import cv2
import logging
import time


if __name__ == "__main__":
    # Set up a video stream
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    
    # Check if we could open the webcam
    if not webcam.isOpened():
        logging.error("Could not open the webcam")
        exit()
    
    while True:
        # Get the next frame
        ret, frame = webcam.read()
        
        # Check if we were able to capture a frame
        if not ret:
            logging.error("Unable to capture the next image")
            break 
        
        # Display the frame
        cv2.imshow('frame', frame)
        
        # If the user presses 'q' the script ends
        if cv2.waitKey(1) == ord('q'):
            break
               