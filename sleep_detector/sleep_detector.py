import cv2
import logging
import argparse

FACE_CASCADE_PATH = "cascade_classifiers/haarcascade_frontalface_alt.xml"
EYE_CASCADE_PATH = "cascade_classifiers/haarcascade_eye_tree_eyeglasses.xml"

if __name__ == "__main__":
     # Logging config
    logging.basicConfig(level=logging.INFO)
    
    # Argument parsing
    parser = argparse.ArgumentParser(
        prog = "sleep_detector",
        description = "A simple script that opens webcam, detects the main face in view and determines if the eyes are open or closed.")
    args = parser.parse_args()
    
    # Set up Haar cascade classifiers (yes, I'm keeping it very basic)
    face_classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    eye_classifier = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    
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
        
        # The image should be in grayscale for the cascade classifiers
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use face cascade classifier to find faces
        faces = face_classifier.detectMultiScale(grayscale_frame, 1.1, 4)
        
        if len(faces):
            # We take the biggest face as the main face
            largest = 0
            main_face = ()
            for (x, y, width, height) in faces:
                if width + height > largest:
                    largest = width + height
                    main_face = (x, y, width, height)
            
            # Process main_face
            (x, y, width, height) = main_face
            
            # Crop the main face region from the image
            face_image = frame[y:y+height, x:x+width]

            # The image should be in grayscale for the cascade classifiers
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Detect the eyes in the face region
            eyes = eye_classifier.detectMultiScale(face_gray, 1.1, 3)
            
            if not len(eyes):
                # Face found, but no eyes
                cv2.putText(frame, "Eyes are closed!", (10, 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
        else:
            # No face found
            cv2.putText(frame, "No face in view", (10, 10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
                        
        
        # Display the frame
        cv2.imshow("frame", frame)
        
        # If the user presses "q" the script ends
        if cv2.waitKey(1) == ord("q"):
            break
               