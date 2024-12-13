import cv2

# Assuming the webcam provides 640x480 resolution
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Resize the frame to the model's input size
    resized_frame = cv2.resize(frame, (640, 384))  # Model expects width=640, height=384

    # (Optional) Display the resized frame
    cv2.imshow("Resized Frame", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
