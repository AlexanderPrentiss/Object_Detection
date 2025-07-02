import cv2
import numpy as np
import serial
import threading
from pypylon import pylon
from pymodbus.client import ModbusTcpClient

# ---------------- CONFIGURATION ---------------- #
SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255
ENCODER_REG = 16388
PISTON_REG = 16383

SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 9600
TICK_INTERVAL = 50  # Trigger every 50 encoder ticks
ENCODER_MAX = 5000  # Wrap-around point

period_reg = 16395
period_val = 200
#period_val = 0

CONTROL_REGISTERS = {
    'auger_rough1': 16383,
    'rough2': 16385,
    'rough3': 16394,
    'belt1': 16386,
    'belt2': 16387,
}

#MOTOR_SPEEDS = {
#    'auger_rough1': 100,
#    'rough2': 110,
#    'rough3': 120,
#    'belt1': 135,
#    'belt2': 100,
#}

MOTOR_SPEEDS = {
    'auger_rough1': 0,
    'rough2': 0,
    'rough3': 0,
    'belt1': 0,
    'belt2': 0,
}


ROI_SLICE = (slice(0, 300), slice(480, 740))  # y range, x range

HSV_RANGES = {
    'belt': [([0, 0, 0], [180, 255, 80])]
}

# ---------------- GLOBAL STATE ---------------- #
latest_frame = None
camera_running = True
last_fired_tick = 0

# ---------------- CAMERA SETUP ---------------- #
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

def camera_loop(camera):
    global latest_frame, camera_running
    while camera_running and camera.IsGrabbing():
        grab_result = camera.RetrieveResult(100, pylon.TimeoutHandling_Return)
        if grab_result is not None and grab_result.IsValid():
            if grab_result.GrabSucceeded():
                latest_frame = grab_result.Array.copy()
            grab_result.Release()

# ---------------- DETECTION ---------------- #
def detect_object_presence(roi_hsv):
    roi_hsv = cv2.GaussianBlur(roi_hsv, (5, 5), 0)

    belt_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES['belt']:
        mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
        belt_mask = cv2.bitwise_or(belt_mask, mask)

    object_mask = cv2.bitwise_not(belt_mask)

    value_channel = roi_hsv[:, :, 2]
    _, bright_mask = cv2.threshold(value_channel, 200, 255, cv2.THRESH_BINARY)
    object_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(bright_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)

    object_pixels = cv2.countNonZero(object_mask)
    return object_pixels > 1000, object_mask

# ---------------- UTILITY ---------------- #
def encoder_diff(current, previous):
    if current >= previous:
        return current - previous
    else:
        return (ENCODER_MAX - previous) + current

# ---------------- MAIN ---------------- #
def main():
    global camera_running, last_fired_tick

    print("[INFO] Initializing components...")
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    camera = init_camera()
    threading.Thread(target=camera_loop, args=(camera,), daemon=True).start()

    last_check_tick = 0
    encoder_value = 0

    if not client.connect():
        print("[ERROR] Could not connect to Modbus server.")
        return

    try:
        print("[INFO] Running detection loop. Press Ctrl+C to exit.")
#        for name, speed in MOTOR_SPEEDS.items():
#           client.write_register(CONTROL_REGISTERS[name], speed, slave=SLAVE_ID)
#
#
#        client.write_register(period_reg, period_val + 1, slave=SLAVE_ID)
#        client.write_register(period_reg, period_val, slave=SLAVE_ID)
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line.isdigit():
                continue

            encoder_value = int(line)
            client.write_register(ENCODER_REG, encoder_value, slave=SLAVE_ID)

            print(f'Encoder Value: {encoder_value}\n')

            if encoder_diff(encoder_value, last_check_tick) >= TICK_INTERVAL and latest_frame is not None:
                last_check_tick = encoder_value
                roi = latest_frame[ROI_SLICE]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                object_present, object_mask = detect_object_presence(roi_hsv)
                object_mask_bgr = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)

                display = roi.copy()
                if object_present:
                    cv2.putText(display, "Object Detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if encoder_diff(encoder_value, last_fired_tick) >= TICK_INTERVAL:
                        fire_tick = (encoder_value + 225) % ENCODER_MAX
                        client.write_register(PISTON_REG, fire_tick, slave=SLAVE_ID)
                        print(f"[ACTION] Fired piston at encoder tick {fire_tick}")
                        last_fired_tick = encoder_value

                combined = np.hstack((display, object_mask_bgr))
                cv2.imshow("Detection (Left: ROI | Right: Object Mask)", combined)

                if cv2.getWindowProperty("Detection (Left: ROI | Right: Object Mask)", cv2.WND_PROP_VISIBLE) < 1:
                    cv2.namedWindow("Detection (Left: ROI | Right: Object Mask)", cv2.WINDOW_NORMAL)

                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")

    finally:
        camera_running = False
        camera.StopGrabbing()
        client.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

