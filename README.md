# Small computer vision apps
Some small programs showcasing computer vision algorithms

All these projects are based on open-cv

## Face detection
An example script that opens the webcam and tries to detect faces using a deep neural network that comes with opencv.
Optionally a command can be given in order to blur all the faces in the video feed, anonymizing the data.

Help for this script:
```
usage: Face Detector [-h] [-p PROTO_FILE] [-m MODEL_FILE] [-c CONFIDENCE] [-b BLURRING] [-d | --display_bbox | --no-display_bbox]

A script that opens webcam and detects faces, optionally it can also blur these faces to anonymize the data.

options:
  -h, --help            show this help message and exit
  -p PROTO_FILE, --protofile PROTO_FILE
                        Prototxt file describing the face detection model.
  -m MODEL_FILE, --modelfile MODEL_FILE
                        Caffe model weights.
  -c CONFIDENCE, --confidence CONFIDENCE
                        Minimum confidence before a detection is processed. Value has to be between 0 and 1.
  -b BLURRING, --blur BLURRING
                        Turn on or off blurring of detected faces.
  -d, --display_bbox, --no-display_bbox
                        Show or hide the bounding box on the video. (default: True)
```
