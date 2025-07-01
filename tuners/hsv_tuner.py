import cv2
import numpy as np
from pypylon import pylon

# === Initialize camera ===
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
cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)

cv2.createTrackbar("H Min", "HSV Tuner", 0, 179, lambda x: None)
cv2.createTrackbar("H Max", "HSV Tuner", 179, 179, lambda x: None)
cv2.createTrackbar("S Min", "HSV Tuner", 0, 255, lambda x: None)
cv2.createTrackbar("S Max", "HSV Tuner", 255, 255, lambda x: None)
cv2.createTrackbar("V Min", "HSV Tuner", 0, 255, lambda x: None)
cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, lambda x: None)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        img = grabResult.Array
        img_resized = cv2.resize(img, (640, 480))

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        # Read sliders
        h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lower, upper)
        mask_bgr = cv2.bitwise_and(img_resized, img_resized, mask=mask)

        # Convert mask to BGR for stacking
        mask_binary_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Stack all 3 feeds horizontally
        stacked = np.hstack((img_resized, mask_bgr, mask_binary_bgr))
        cv2.imshow("HSV Tuner", stacked)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Selected HSV Range: lower={lower}, upper={upper}")
            break

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()