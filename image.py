import cv2

class ImageCapture:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video camera.")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error: Can't receive frame (stream end?).")
        return frame

    def release(self):
        self.cap.release()
