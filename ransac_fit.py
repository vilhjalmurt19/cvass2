import numpy as np

class LineFit:
    def __init__(self, max_iter=1000, thresh=1.0, min_in=100):
        self.max_iter = max_iter
        self.thresh = thresh
        self.min_in = min_in

    def fit(self, edges):
        if len(edges) < 2:
            return None, None

        best = None
        best_count = 0

        for _ in range(self.max_iter):
            idx = np.random.choice(len(edges), 2, replace=False)
            p1, p2 = edges[idx]

            if p2[0] == p1[0]:
                continue
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = p1[1] - m * p1[0]

            inliers = []
            for p in edges:
                x, y = p
                y_est = m * x + b
                if abs(y - y_est) < self.thresh:
                    inliers.append(p)

            if len(inliers) > best_count and len(inliers) >= self.min_in:
                best_count = len(inliers)
                best = (m, b)

        if best is None:
            return None, None

        x_min, x_max = edges[:, 0].min(), edges[:, 0].max()
        y_min = best[0] * x_min + best[1]
        y_max = best[0] * x_max + best[1]

        return np.array([x_min, x_max]), np.array([y_min, y_max])
