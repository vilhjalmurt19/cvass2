import cv2
import numpy as np

class CannyEdgeDetector:
    def __init__(self, low=100, high=200):
        self.low = low
        self.high = high

    def apply(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.low, self.high)
        return edges

    def get_coords(self, edges):
        y, x = np.nonzero(edges)
        coords = np.column_stack((x, y))
        return coords
