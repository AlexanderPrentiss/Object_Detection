import cv2
import numpy as np
from pypylon import pylon
from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()  # id → centroid
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects

def init_camera():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.Width.SetValue(camera.Width.Max)
    camera.Height.SetValue(camera.Height.Max)
    camera.PixelFormat.SetValue("BGR8")
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(8000)
    camera.GainAuto.SetValue("Continuous")
    camera.AcquisitionFrameRateEnable.SetValue(True)
    camera.AcquisitionFrameRate.SetValue(120)
    camera.BalanceWhiteAuto.SetValue("Continuous")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camera

def resize(img, size=(640, 480)):
    return cv2.resize(img, size)

ROI = (392, 784, 704, 920) # x_min, x_max, y_min, y_max

camera = init_camera()

backSub = cv2.createBackgroundSubtractorMOG2(history=600)

median_background = np.load("background_median.npy")

tracker = CentroidTracker(max_disappeared=10, max_distance=50)

try:
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            img = grabResult.Array

            blurred = cv2.GaussianBlur(img, (5, 5), 0)

            fg_mask = backSub.apply(blurred)
            mask_retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

            diff = cv2.absdiff(blurred, median_background)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            med_retval, med_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

            # final_mask = cv2.bitwise_and(mask_thresh, med_thresh)
            final_mask = med_thresh

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel) # Remove noise
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel) # Fill holes

            contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_contour_area = 500
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            frame_ct = cv2.drawContours(img, large_contours, -1, (0, 255, 0), 2)

            centroids = []
            for cnt in large_contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    # # Draw the centroid
                    # cv2.circle(frame_ct, (cx, cy), 5, (0, 0, 255), -1)
                    # # Optionally label it
                    # cv2.putText(frame_ct, f"({cx},{cy})", (cx + 10, cy),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            objects = tracker.update(centroids)

            for object_id, centroid in objects.items():
                cv2.circle(frame_ct, centroid, 5, (0, 255, 255), -1)
                cv2.putText(frame_ct, f"ID {object_id}", (centroid[0]+10, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Convert grayscale masks to 3-channel BGR
            mask_thresh_color = cv2.cvtColor(mask_thresh, cv2.COLOR_GRAY2BGR)
            med_thresh_color = cv2.cvtColor(med_thresh, cv2.COLOR_GRAY2BGR)

            # Resize everything uniformly
            top_row = np.hstack([
                resize(img),  # already BGR
                resize(mask_thresh_color)
            ])
            bottom_row = np.hstack([
                resize(med_thresh_color),
                resize(frame_ct)
            ])
            combined = np.vstack([top_row, bottom_row])

            cv2.imshow("Combined View", combined)
            cv2.imshow("Final frame", frame_ct)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        grabResult.Release()

finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()