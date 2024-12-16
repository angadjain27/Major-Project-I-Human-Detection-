import cv2
import numpy as np
from ultralytics import YOLO
import math

class HumanDetector: 
    def __init__(self, camera_height=10.0, camera_angle=45.0, focal_length=1000):
        self.model = YOLO('yolov8n.pt')
        self.camera_height = camera_height  # meters
        self.camera_angle = math.radians(camera_angle)
        self.focal_length = focal_length
        
    def estimate_height_from_ground(self, y_pos, frame_height):
        # Convert pixel position to angle from camera
        angle_in_frame = math.atan2((frame_height/2 - y_pos), self.focal_length)
        total_angle = self.camera_angle + angle_in_frame
        
        # Calculate distance using trigonometry
        distance = self.camera_height / math.tan(total_angle)
        return distance

    def process_video(self, source=0):
        cap = cv2.VideoCapture(source)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLOv8 detection
            results = self.model.predict(frame, classes=[0])  # class 0 is person
            
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate distance from camera
                    distance = self.estimate_height_from_ground(y2, frame.shape[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Display distance
                    label = f"Distance: {distance:.2f}m"
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Human Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize detector with camera parameters
    detector = HumanDetector(
        camera_height=10.0,  # Adjust based on your camera height in meters
        camera_angle=45.0,   # Adjust based on your camera angle in degrees
        focal_length=1000    # Adjust based on your camera's focal length
    )
    
    # For webcam, use 0
    # For video file, use the path: "path/to/your/video.mp4"
    detector.process_video(0)