import cv2
import numpy as np
from pypylon import pylon

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

def capture_background_median(camera, num_frames):
    frame_list = []
    print(f"[INFO] Capturing {num_frames} frames for background median computation...")

    while camera.IsGrabbing() and len(frame_list) < num_frames:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            frame_list.append(blurred)
        grabResult.Release()

    print("[INFO] Computing median...")
    median_img = np.median(np.stack(frame_list, axis=0), axis=0).astype(np.uint8)
    np.save("background_median.npy", median_img)
    print("[INFO] Median background image saved as 'background_median.npy'.")

    return median_img

camera = init_camera()

try:
    background_median = capture_background_median(camera, num_frames=1000)
    cv2.imshow("Background Median", background_median)
    print("[INFO] Press any key to close...")
    cv2.waitKey(0)

finally:
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

