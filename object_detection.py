import cv2
import numpy as np
from pypylon import pylon

# HSV range for the belt (everything else is considered "object")
HSV_RANGES = {
    'belt': [
        ([0, 0, 0], [180, 255, 60])  # Adjust for your actual belt color
    ]
}

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
    camera.AcquisitionFrameRate.SetValue(60)
    camera.BalanceWhiteAuto.SetValue("Continuous")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camera

def detect_object_presence(roi_hsv):
    roi_hsv = cv2.GaussianBlur(roi_hsv, (5, 5), 0)
    belt_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES['belt']:
        mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
        belt_mask = cv2.bitwise_or(belt_mask, mask)

    object_mask = cv2.bitwise_not(belt_mask)
    object_pixels = cv2.countNonZero(object_mask)

    return object_pixels > 1000, object_mask

if __name__ == '__main__':
    camera = init_camera()
    print("[INFO] Starting object detection test. Press 'q' to exit.")

    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = grab_result.Array
                roi = frame[0:400, 100:700]  # Modify ROI as needed
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                object_present, object_mask = detect_object_presence(roi_hsv)

                # Convert binary mask to BGR for visualization
                object_mask_bgr = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)

                # Annotate ROI if object is present
                display = roi.copy()
                if object_present:
                    cv2.putText(display, "Object Detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Combine ROI and mask horizontally
                combined = np.hstack((display, object_mask_bgr))

                # Show in one window
                cv2.imshow("Detection (Left: ROI | Right: Object Mask)", combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            grab_result.Release()

    finally:
        camera.StopGrabbing()
        cv2.destroyAllWindows()

