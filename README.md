# Animal-Behavior-Tracking-System

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
   - Download or train a YOLO model suitable for animal detection.
     this should include coco.names , yolo
   
3. **Run the System:**
   - Input your video streams and analyze the outputs for behavior tracking and injury detection.

