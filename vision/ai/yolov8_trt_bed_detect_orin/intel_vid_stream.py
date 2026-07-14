import cv2

camera = cv2.VideoCapture(6)

if not camera.isOpened():
    print("Could not open camera")
    raise SystemExit(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("RealSense D435i", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()