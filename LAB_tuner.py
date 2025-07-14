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
cv2.namedWindow("LAB Tuner", cv2.WINDOW_NORMAL)

cv2.createTrackbar("L Min", "LAB Tuner", 0, 255, lambda x: None)
cv2.createTrackbar("L Max", "LAB Tuner", 255, 255, lambda x: None)
cv2.createTrackbar("A Min", "LAB Tuner", 0, 255, lambda x: None)
cv2.createTrackbar("A Max", "LAB Tuner", 255, 255, lambda x: None)
cv2.createTrackbar("B Min", "LAB Tuner", 0, 255, lambda x: None)
cv2.createTrackbar("B Max", "LAB Tuner", 255, 255, lambda x: None)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        img = grabResult.Array
        img_resized = cv2.resize(img, (640, 480))

        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)

        # Read sliders
        l_min = cv2.getTrackbarPos("L Min", "LAB Tuner")
        l_max = cv2.getTrackbarPos("L Max", "LAB Tuner")
        a_min = cv2.getTrackbarPos("A Min", "LAB Tuner")
        a_max = cv2.getTrackbarPos("A Max", "LAB Tuner")
        b_min = cv2.getTrackbarPos("B Min", "LAB Tuner")
        b_max = cv2.getTrackbarPos("B Max", "LAB Tuner")

        lower = np.array([l_min, a_min, b_min])
        upper = np.array([l_max, a_max, b_max])

        mask = cv2.inRange(lab, lower, upper)
        masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        stacked = np.hstack((img_resized, masked, mask_bgr))
        cv2.imshow("LAB Tuner", stacked)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Selected LAB Range: lower={lower}, upper={upper}")
            break

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
