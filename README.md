# Small computer vision apps
Some small programs showcasing computer vision algorithms

All these projects are based on open-cv

## Face detection
An example script that opens the webcam and tries to detect faces using a deep neural network that comes with opencv.
Optionally a command can be given in order to blur all the faces in the video feed, anonymizing the data.

<img src="https://github.com/VerleysenNiels/small_computer_vision_apps/blob/master/face_detector/examples/face_detect.gif?raw=true" height="200"> <img src="https://github.com/VerleysenNiels/small_computer_vision_apps/blob/master/face_detector/examples/face_blur.gif?raw=true" height="200">

Help for this script:
```
usage: Face Detector [-h] [-p PROTO_FILE] [-m MODEL_FILE] [-c CONFIDENCE] [-b | --blur | --no-blur] [-d | --display_bbox | --no-display_bbox]

A script that opens webcam and detects faces, optionally it can also blur these faces to anonymize the data.

options:
  -h, --help            show this help message and exit
  -p PROTO_FILE, --protofile PROTO_FILE
                        Prototxt file describing the face detection model.
  -m MODEL_FILE, --modelfile MODEL_FILE
                        Caffe model weights.
  -c CONFIDENCE, --confidence CONFIDENCE
                        Minimum confidence before a detection is processed. Value has to be between 0 and 1.
  -b, --blur, --no-blur
                        Turn on or off blurring of detected faces. (default: False)
  -d, --display_bbox, --no-display_bbox
                        Show or hide the bounding box on the video. (default: True)
```

## Document scanner
A script that turns your camera into a document scanner. Pass an image of a document to the script and it will transform it as if you would have put it in a scanner.

<img src="https://github.com/VerleysenNiels/small_computer_vision_apps/blob/master/document_scanner/images/result.PNG?raw=true" height="300">

```
usage: Document Scanner [-h] [-i IMAGE_PATH] [-r | --recolor | --no-recolor] [-s | --show_steps | --no-show_steps]

A script that takes a picture of a document and transforms it as if you would have scanned the document.

options:
  -h, --help            show this help message and exit
  -i IMAGE_PATH, --image IMAGE_PATH
                        Image of the document to be scanned.
  -r, --recolor, --no-recolor
                        Turn on or off recoloring of the scanned document. (default: False)
  -s, --show_steps, --no-show_steps
                        Show the different steps taken by the scanner. (default: False)
```

## Object tracker
A script that turns your camera into a simple object tracker. Tracking happens based on the configured colorrange of the object.

<img src="https://github.com/VerleysenNiels/small_computer_vision_apps/blob/master/object_tracker/tracking.gif?raw=true" height="300">

```
usage: object_tracker [-h] [-i MIN_COLOR] [-a MAX_COLOR] [-l TRAIL_LENGTH] [-c TRAIL_COLOR]

A script that uses the webcam to detects and track an object, based on the color.

options:
  -h, --help            show this help message and exit
  -i MIN_COLOR, --mincolor MIN_COLOR
                        First HSV value of the minimum color, the object with a color between min and max will be
                        tracked.
  -a MAX_COLOR, --maxcolor MAX_COLOR
                        First HSV value of the maximum color, the object with a color between min and max will be
                        tracked.
  -l TRAIL_LENGTH, --traillength TRAIL_LENGTH
                        Length of the trail to be drawn on the image. Pass zero as argument to turn off the trail.
  -c TRAIL_COLOR, --trailcolor TRAIL_COLOR
                        Color of the trail, can be either blue, green or red.
```
