# Human Detection and Height Estimation System

This system detects humans in video feeds from CCTV or drone cameras and estimates their distance from the camera.

## Features

- Real-time human detection using YOLOv8
- Distance estimation from camera
- Works with both live camera feed and video files
- Optimized for aerial/CCTV views

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the detector:
```bash
python human_detection.py
```

## Configuration

Adjust these parameters in the code based on your setup:

- `camera_height`: Height of camera from ground (meters)
- `camera_angle`: Angle of camera from horizontal (degrees)
- `focal_length`: Camera focal length (pixels)

## Usage

- Press 'q' to quit the application
- The green boxes show detected humans
- Distance measurements are shown above each detection