import torch
import cv2
import numpy as np
from pymavlink import mavutil

# Load the YOLOv5 model using torch.hub
model = torch.hub.load("ultralytics/yolov5", "custom", path="path/to/best (3).pt")

# Initialize the video
cap = cv2.VideoCapture('output.mp4')

# Check if the video is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Connect to the MAVLink telemetry stream
mavlink_connection = mavutil.mavlink_connection('udp:localhost:14550')

# Define object-specific variables for distance measurement
KNOWN_WIDTH = 5.0  # Known width of the object (m)
focal_length = 0.0315  # Pre-calculated focal length in meters (adjust based on your camera calibration)

# Function to calculate distance
def get_distance(known_width, focal_length, per_width):
    return (known_width * focal_length) / per_width

# Function to get the current altitude of the aircraft
def get_current_altitude(mavlink_connection):
    # Wait for a message with a timeout
    msg = mavlink_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    if msg is not None:
        return msg.relative_alt / 1000.0  # Convert from mm to meters
    return None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Update KNOWN_DISTANCE based on current altitude
    altitude = get_current_altitude(mavlink_connection)
    if altitude is not None:
        KNOWN_DISTANCE = altitude

    # Convert the frame to a format compatible with YOLOv5
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img)

    # Render results on the frame
    results.render()

    # Extract bounding boxes from YOLOv5 results
    boxes = results.xyxy[0].tolist()

    # Loop through each bounding box
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        width_in_pixels = x2 - x1

        # Calculate distance using the bounding box width
        distance = get_distance(KNOWN_WIDTH, focal_length, width_in_pixels)

        # Draw a rectangle on the contour
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

        # Write the distance on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int(x1), int(y1) - 10)
        fontScale = 0.6
        color = (0, 0, 255)
        thickness = 2
        cv2.putText(frame, f'Distance: {distance:.2f} m', org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close the window
cap.release()
cv2.destroyAllWindows()
