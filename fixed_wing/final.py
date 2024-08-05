import cv2
import torch
import RPi.GPIO as GPIO
import time

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="path/to/best.pt")


width = 8000 # Define object-specific variables for distance measurement
KNOWN_WIDTH = 500     # Known width of the object (meters)
focal_length = 1.64  # Pre-calculated focal length (adjust based on your camera calibration)

# Setup Servo
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
servo = GPIO.PWM(11, 50)
servo.start(0)

def set_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(11, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(11, False)
    servo.ChangeDutyCycle(0)

# Start video capture
# cap = cv2.VideoCapture(0)
# Initialize the webcam
cap = cv2.VideoCapture('output.mp4')

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Garis-garis untuk panduan visual
    line1 = cv2.line(frame, (200, 0), (200, height), (255, 0, 0), 1)
    line2 = cv2.line(frame, (440, 0), (440, height), (255, 0, 0), 1)
    line3 = cv2.line(frame, (0, 180), (width, 180), (255, 0, 0), 1)
    line4 = cv2.line(frame, (0, 300), (width, 300), (255, 0, 0), 1)

    # Convert frame to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects using YOLOv5
    results = model(img_rgb)

    def get_dist(rectangle_params, image):
        # Find the number of pixels covered
        pixels = rectangle_params[1][0]
        print(pixels)
        # Calculate distance
        dist = (width * focal_length) / pixels
        # Write the distance on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (0, 20)
        fontScale = 0.6
        color = (0, 0, 255)
        thickness = 2
        image = cv2.putText(image, 'Distance from Camera in CM:', org, font, 1, color, 2, cv2.LINE_AA)
        image = cv2.putText(image, str(dist), (110, 50), font, fontScale, color, 1, cv2.LINE_AA)
        return image

    # Process the results
    for det in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            centerX = (x1 + x2) // 2
            centerY = (y1 + y2) // 2

            # Draw bounding box and center of the object
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 3)
            frame = cv2.circle(frame, (centerX, centerY), 5, (0, 255, 0), -1)

            # Calculate offset from the center of the frame
            frame_center_x = width // 2
            frame_center_y = height // 2
            offset_x = centerX - frame_center_x
            offset_y = centerY - frame_center_y

            # Determine action based on the offset
            if abs(offset_x) < 20 and abs(offset_y) < 20:
                cv2.putText(frame, 'CENTERED: DROPPING PACKAGE', (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                set_angle(180)  # Move servo to 180 degrees
            else:
                if offset_x > 20:
                    cv2.putText(frame, 'Move Left', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    # Send command to Pixhawk to move left
                elif offset_x < -20:
                    cv2.putText(frame, 'Move Right', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    # Send command to Pixhawk to move right
                if offset_y > 20:
                    cv2.putText(frame, 'Move Up', (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    # Send command to Pixhawk to move up
                elif offset_y < -20:
                    cv2.putText(frame, 'Move Down', (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    # Send command to Pixhawk to move down

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

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
servo.stop()
GPIO.cleanup()
cv2.destroyAllWindows()