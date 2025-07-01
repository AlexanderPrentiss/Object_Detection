import cv2
import numpy as np
from pypylon import pylon

# === Camera initialization ===
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

camera = init_camera()

# === Create window and sliders ===
cv2.namedWindow("ROI Tuner", cv2.WINDOW_NORMAL)

# We'll wait for first frame to get image dimensions
first_frame = True
frame_width, frame_height = 640, 480  # default fallback

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        img = grabResult.Array

        # Get true image size on first frame
        if first_frame:
            frame_height, frame_width = img.shape[:2]
            # Create sliders once we know dimensions
            cv2.createTrackbar("Y Min", "ROI Tuner", 0, frame_height, lambda x: None)
            cv2.createTrackbar("Y Max", "ROI Tuner", frame_height, frame_height, lambda x: None)
            cv2.createTrackbar("X Min", "ROI Tuner", 0, frame_width, lambda x: None)
            cv2.createTrackbar("X Max", "ROI Tuner", frame_width, frame_width, lambda x: None)
            first_frame = False

        # Get slider values
        y_min = cv2.getTrackbarPos("Y Min", "ROI Tuner")
        y_max = cv2.getTrackbarPos("Y Max", "ROI Tuner")
        x_min = cv2.getTrackbarPos("X Min", "ROI Tuner")
        x_max = cv2.getTrackbarPos("X Max", "ROI Tuner")

        # Safety: ensure min <= max
        if y_min > y_max:
            y_max = y_min
            cv2.setTrackbarPos("Y Max", "ROI Tuner", y_max)
        if x_min > x_max:
            x_max = x_min
            cv2.setTrackbarPos("X Max", "ROI Tuner", x_max)

        # Draw ROI rectangle on live frame
        preview = img.copy()
        cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        cv2.imshow("ROI Tuner", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Final ROI: X [{x_min}, {x_max}], Y [{y_min}, {y_max}]")
            break

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()