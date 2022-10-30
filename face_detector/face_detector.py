import cv2
import logging
import numpy as np
import time

# Temporarily have these hardcoded, later add them as arguments
PROTO_FILE = "model/deploy.prototxt.txt"
MODEL_FILE = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
MIN_CONFIDENCE = 0.6
BLURRING = True

if __name__ == "__main__":
    # Logging config
    logging.basicConfig(level=logging.INFO)
    
    # Load in the face detector neural network that is included with 
    network = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)
    
    # Set up a video stream
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    
    ret, frame = webcam.read()
    
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
        
        # My camera has frames with shape (1280, 720, 3)
        (height, width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (640, 360)), 1.0, (300, 300), (104, 177, 123))
        
        # Make predictions
        network.setInput(blob)
        detections = network.forward()
     
        # Process the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Don't look at detections with only a low confidence
            if confidence < MIN_CONFIDENCE:
                continue
            
            # Get the bounding box
            bbox = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = bbox.astype("int")
            
            # Blurring crashes on the negative numbers that sometimes come out of the prediction
            startX = 0 if startX < 0 else startX
            startY = 0 if startY < 0 else startY
            endX = 0 if endX < 0 else endX
            endY = 0 if endY < 0 else endY            
            
            if BLURRING:
                # Add blurring over the faces
                face = frame[startY:endY, startX:endX]
                face = cv2.blur(face, (55, 55))
                frame[startY:startY+face.shape[0], startX:startX+face.shape[1]] = face
                
            # Draw the bounding box on the frame
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('frame', frame)
        
        # If the user presses 'q' the script ends
        if cv2.waitKey(1) == ord('q'):
            break
               