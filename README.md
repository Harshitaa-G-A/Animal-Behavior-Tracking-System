# Animal Behavior Tracking System

The Animal Behavior Tracking System utilizes advanced computer vision techniques to monitor and analyze animal behavior in video footage.

## Features

- **Object Detection with YOLO:** Utilizes a robust YOLO model for accurate animal detection in input video streams.
  
- **Mouth Region Isolation:** Implements a Haar Cascade classifier system to isolate the mouth region of detected animals for further analysis.
  
- **Color Analysis for Injury Detection:** Uses color analysis to identify potential injuries in animals. By analyzing color variations, particularly red tones, abnormalities that may indicate injuries or stress can be identified.
  
- **Optical Flow Analysis:** Applies optical flow analysis to consecutive video frames to detect motion patterns indicating running or resting behavior. This analysis is performed using OpenCV for video processing and frame manipulation.
  
- **Threshold Technique:** Utilizes a thresholding technique to determine whether the animal's movement exceeds predefined thresholds, indicating running or resting behavior.

## Getting Started

To get started with the Animal Behavior Tracking System:

1. **Setup Environment:**
   - Install necessary dependencies, including OpenCV and YOLO.

2. **Configure YOLO Model:**
   - Download or train a YOLO model suitable for animal detection. You will need:
     - [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)
     - [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
     - [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

3. **Run the System:**
   - Input your video streams and analyze the outputs for behavior tracking and injury detection.

## Tracked Output

### Sample Output Images

![Image 1](https://github.com/Harshitaa-G-A/Animal-Behavior-Tracking-System/assets/146211436/a2f413c5-2c93-4936-ab8b-8fea3ee1fdab)

![Image 2](https://github.com/Harshitaa-G-A/Animal-Behavior-Tracking-System/assets/146211436/563a1471-eeed-4bbb-ae30-0da0d905e6e6)

![Image 3](https://github.com/Harshitaa-G-A/Animal-Behavior-Tracking-System/assets/146211436/62912382-bd31-4946-b664-68105ae15acc)

![Image 4](https://github.com/Harshitaa-G-A/Animal-Behavior-Tracking-System/assets/146211436/6f281562-99f2-45ec-a73b-b32f57972c88)

![Image 5](https://github.com/Harshitaa-G-A/Animal-Behavior-Tracking-System/assets/146211436/2549033e-22e3-4db4-92f4-fe63fce58035)
