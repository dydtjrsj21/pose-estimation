import cv2

def list_available_webcams():
    index = 0
    available_cameras = []

    while True:
        # Try to open the camera
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        else:
            available_cameras.append(index)
            cap.release()
        index += 1

    return available_cameras

if __name__ == "__main__":
    webcams = list_available_webcams()
    if webcams:
        print("Available webcams:")
        for cam in webcams:
            print(f"Camera index: {cam}")
    else:
        print("No webcams found.")
