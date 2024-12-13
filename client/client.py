import cv2
import asyncio
import websockets
import numpy as np
import json
import time

async def stream_video():
    async with websockets.connect("ws://210.126.11.81:8765") as websocket:
        cap = cv2.VideoCapture(0)  # Use the first webcam

        burpee_count = 0
        timer_start = None
        timer_active = False
        threshold_close = 50  # Threshold for nose and ankle proximity
        threshold_far = 200  # Threshold for nose and ankle distance
        min_time = 3  # Minimum time in seconds for a valid burpee
        prev_ankle_y = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send(buffer.tobytes())

            # Receive keypoints data from the server
            data = await websocket.recv()
            try:
                keypoints_data = json.loads(data)  # Parse JSON
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                continue

            # Define skeleton connections
            skeleton = [
                ("nose", "left_eye"), ("nose", "right_eye"),
                ("left_eye", "left_ear"), ("right_eye", "right_ear"),
                ("nose", "left_shoulder"), ("nose", "right_shoulder"),
                ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
                ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
                ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
                ("left_hip", "left_knee"), ("right_hip", "right_knee"),
                ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
                ("left_hip", "right_hip")
            ]

            # Draw keypoints and skeleton on the frame
            for keypoints in keypoints_data:
                points = {}
                for key, value in keypoints.items():
                    x, y, confidence = value["x"], value["y"], value["confidence"]
                    if confidence > 0.5:  # Only draw keypoints with high confidence
                        points[key] = (int(x), int(y))
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

                # Draw skeleton lines
                for start, end in skeleton:
                    if start in points and end in points:
                        cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

                # Burpee detection logic
                if "nose" in points:
                    nose_y = points["nose"][1]
                    ankle_y = prev_ankle_y

                    if "left_ankle" in points and "right_ankle" in points:
                        ankle_y = min(points["left_ankle"][1], points["right_ankle"][1])
                        prev_ankle_y = min(points["left_ankle"][1], points["right_ankle"][1])
                    

                    height_diff = abs(nose_y - ankle_y)

                    if height_diff < threshold_close:
                        if not timer_active:
                            timer_start = time.time()
                            timer_active = True
                    elif height_diff > threshold_far:
                        if timer_active:
                            timer_end = time.time()
                            timer_active = False

                            if timer_start and (timer_end - timer_start) >= min_time:
                                burpee_count += 1


                # Display burpee count
                cv2.putText(frame, f"Burpee Count: {burpee_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display the frame
            cv2.imshow("Pose Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

asyncio.run(stream_video())
