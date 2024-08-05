import torch
import cv2
import numpy as np
import asyncio
from mavsdk import System
from mavsdk.action import ActionError

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Known width of the object (e.g., 20 cm)
KNOWN_WIDTH = 20.0

# Focal length (calculated from a calibration step)
FOCAL_LENGTH = 800.0  # Example value, this needs to be calibrated

async def connect_plane():
    plane = System()
    await plane.connect(system_address="udp://:14540")

    async for state in plane.core.connection_state():
        if state.is_connected:
            print(f"Plane connected: {state.uuid}")
            break

    return plane

async def set_servo_angle(plane, channel, angle):
    pwm_value = int(1000 + (angle / 180) * 1000)
    try:
        await plane.action.set_servo(channel, pwm_value)
    except ActionError as e:
        print(f"Failed to set servo: {e}")

def calculate_distance(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width

async def main():
    plane = await connect_plane()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        for x1, y1, x2, y2, conf, cls in detections:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            perceived_width = int(x2 - x1)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Calculate distance to the object
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, perceived_width)
            print(f'Distance to object: {distance} cm')

            # Adjust servo to center object
            frame_center_x = frame.shape[1] // 2
            error = frame_center_x - cx
            angle = error * 0.1  # Adjust this value based on your system
            await set_servo_angle(plane, 1, angle)

        cv2.imshow('YOLOv5 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    asyncio.run(main())
