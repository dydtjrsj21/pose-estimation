import cv2
import numpy as np
import asyncio
import json
import websockets
from furiosa.models.vision import YOLOv7w6Pose
from furiosa.runtime.sync import create_runner

async def handle_client(websocket, path):
    yolo_pose = YOLOv7w6Pose()

    with create_runner(yolo_pose.model_source()) as runner:
        while True:
            # Receive image from the client
            data = await websocket.recv()
            np_data = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            # Preprocess and run pose estimation
            inputs, contexts = yolo_pose.preprocess([image])
            output = runner.run(inputs)
            results = yolo_pose.postprocess(output, contexts=contexts)

            # Extract keypoints and send back to the client
            keypoints_data = []
            for result_list in results:  # Iterate through the list of results
                for result in result_list:  # Iterate through each result
                    keypoints = {
                        key: {
                            "x": getattr(result, key).x,
                            "y": getattr(result, key).y,
                            "confidence": getattr(result, key).confidence,
                        }
                        for key in [
                            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist", "left_hip", "right_hip",
                            "left_knee", "right_knee", "left_ankle", "right_ankle"
                        ]
                    }
                    keypoints_data.append(keypoints)

            await websocket.send(json.dumps(keypoints_data))

# Start the WebSocket server
start_server = websockets.serve(handle_client, "0.0.0.0", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
