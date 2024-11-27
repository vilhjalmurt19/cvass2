import time
import cv2
from image import ImageCapture
from canny_detector import CannyEdgeDetector
from ransac_fit import LineFit

img = ImageCapture()
edge = CannyEdgeDetector(low=20, high=100)
line = LineFit(max_iter=50, thresh=1.0, min_in=75)

try:
    total_time = 0
    frame_count = 0

    while True:
        start = time.time()

        frame = img.capture_frame()
        edges = edge.apply(frame)
        coords = edge.get_coords(edges)
        x, y = line.fit(coords)

        if x is not None and y is not None:
            cv2.line(frame, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 255, 0), 2)

        cv2.imshow('Frame with Line', frame)
        cv2.imshow('Edges', edges)

        elapsed = time.time() - start
        total_time += elapsed
        frame_count += 1

        print(f"Processing time: {elapsed:.4f} seconds")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if frame_count > 0:
        avg_time = total_time / frame_count
        print(f"Average processing time per frame: {avg_time:.4f} seconds")
    img.release()
    cv2.destroyAllWindows()
