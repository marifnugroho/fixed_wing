import torch
import cv2
import time
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define object specific variables for distance measurement
dist = 0
focal = 450
pixels = 30
width = 4

# Function to calculate distance
def get_dist(rectangle_params, image):
    # Find the number of pixels covered
    pixels = rectangle_params[1][0]
    print(pixels)
    # Calculate distance
    dist = (width * focal) / pixels
    # Write the distance on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (0, 20)
    fontScale = 0.6
    color = (0, 0, 255)
    thickness = 2
    image = cv2.putText(image, 'Distance from Camera in CM:', org, font, 1, color, 2, cv2.LINE_AA)
    image = cv2.putText(image, str(dist), (110, 50), font, fontScale, color, 1, cv2.LINE_AA)
    return image

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to a format compatible with YOLOv5
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img)

    # Render results on the frame
    results.render()

    # Convert the image back to BGR for OpenCV
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Extract bounding boxes from YOLOv5 results
    boxes = results.xyxy[0].tolist()

    # Loop through each bounding box
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        # Draw a rectangle on the contour
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

        # Calculate distance using the bounding box dimensions
        rectangle_params = ((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), 0
        frame = get_dist(rectangle_params, frame)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()