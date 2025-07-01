import cv2
import numpy as np
from pypylon import pylon
from pymodbus.client import ModbusTcpClient

# ---------------- CONFIGURATION SECTION ---------------- #

# Camera ROI (crop window)
# ROI = (392, 784, 704, 920) # x_min, x_max, y_min, y_max
ROI = (642, 1024, 345, 859) # x_min, x_max, y_min, y_max

# HSV color ranges
# HSV color ranges tuned for shiny lighting
HSV_RANGES = {
    'blue':   [[90, 80, 100], [140, 255, 255]],
    'red':    ([[0, 80, 100], [10, 255, 255]], [[160, 80, 100], [179, 255, 255]]),
    'green':  [[40, 60, 100], [85, 255, 255]],
    'gold':   [[15, 80, 120], [45, 255, 255]],
    'silver': [[0, 0, 100], [179, 50, 220]],
    'belt':   [[0, 0, 0], [179, 100, 80]],
}

# Encoder offset per piston (how many ticks away they are from camera gate)
PISTON_ENCODER_OFFSETS = {
    'blue': 15,
    'red': 15,
    'green': 15,
    'orange': 15,
    'black': 15
}

# Modbus register mappings for 5 pistons
PISTON_REGISTERS = {
    'blue': 16383,
    'red': 16384,
    'green': 16385,
    'orange': 16386,
    'black': 16387
}

# General system settings
GRACE_FRAMES = 0
ENCODER_REGISTER = 24575
SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255

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
    camera.AcquisitionFrameRate.SetValue(120)
    camera.BalanceWhiteAuto.SetValue("Continuous")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camera

# ---------------- COLOR CLASSIFICATION ---------------- #

def classify_color(roi_hsv):
    belt_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES['belt']:
        mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
        belt_mask = cv2.bitwise_or(belt_mask, mask)

    non_belt_mask = cv2.bitwise_not(belt_mask)

    best_color = None
    best_count = 0

    for color, ranges in HSV_RANGES.items():
        if color == 'belt':
            continue

        mask_total = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
            mask_total = cv2.bitwise_or(mask_total, mask)

        mask_total = cv2.bitwise_and(mask_total, non_belt_mask)
        count = cv2.countNonZero(mask_total)

        if count > best_count:
            best_count = count
            best_color = color

    return best_color if best_count > 1500 else None

# ---------------- MAIN CONTROL LOGIC ---------------- #

if __name__ == '__main__':
    # Initialize Modbus and camera
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)
    camera = init_camera()

    # Create separate queues for each piston
    piston_queues = {color: [] for color in PISTON_REGISTERS.keys()}


    last_detection = None
    grace_counter = 0
    index = 0
    
    last_fire_values = {color: float('-inf') for color in PISTON_REGISTERS.keys()}

    if client.connect():
        print("Connected to Modbus server")

        try:
            while camera.IsGrabbing():
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    img = grabResult.Array
                    roi = img[ROI[2]:ROI[3], ROI[0]:ROI[1]]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # Get current encoder reading
                    encoder_value = client.read_input_registers(ENCODER_REGISTER, slave=SLAVE_ID).registers[0]

                    # Handle new detection with grace frame logic
                    detected_color = None
                    if grace_counter == 0:
                        current_detection = classify_color(hsv)
                        print(current_detection)
                        if current_detection and current_detection != last_detection:
                            detected_color = current_detection
                            grace_counter = GRACE_FRAMES

                            # Compute target encoder tick for piston
                            target_tick = encoder_value + PISTON_ENCODER_OFFSETS[current_detection]
                            piston_queues[current_detection].append(target_tick)
                            print(f"{index} | Detected: {current_detection} | Scheduled for tick: {target_tick}")
                            index += 1
                        else:
                            detected_color = None
                        last_detection = current_detection
                    else:
                        grace_counter -= 1

                    # Check if any piston queue has ready-to-fire objects
                    for color, queue in piston_queues.items():
                        if encoder_value > last_fire_values[color] and queue:
                            next_fire_value = queue.pop(0)
                            register = PISTON_REGISTERS[color]
                            print("color\n")
                            client.write_register(register, next_fire_value, slave=SLAVE_ID)
                            last_fire_values[color] = next_fire_value
                            print(f"Firing {color} piston at encoder {next_fire_value}")

                    # Overlay text for debugging
                    text = f"{index} | {last_detection if last_detection else 'Waiting...'}"
                    cv2.putText(roi, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Scrap Detector", roi)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                grabResult.Release()

        except KeyboardInterrupt:
            print("Stopping system...")
        finally:
            client.close()
            camera.StopGrabbing()
            camera.Close()
            cv2.destroyAllWindows()
    else:
        print("Failed to connect to Modbus server")
