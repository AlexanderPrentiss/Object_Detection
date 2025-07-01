import cv2
import serial
import numpy as np
from pypylon import pylon
from pymodbus.client import ModbusTcpClient 
import time

# ---------------- CONFIGURATION SECTION ---------------- #
ROI = (375, 675, 0, 100) 

HSV_RANGES = {
    'blue':   [([90, 120, 100], [130, 255, 255])],
    'red':    [([0, 150, 120], [10, 255, 255]), ([165, 150, 120], [179, 255, 255])],
    'green':  [([50, 100, 100], [85, 255, 255])],
    'gold':   [([15, 120, 130], [40, 255, 255])],
    'silver': [([0, 0, 90], [179, 25, 200])],
    'belt':   [([0, 0, 135], [179, 100, 255])]
}

PISTON_ENCODER_OFFSETS = {
    'blue': 110,
    'red': 190,
    'green': 270,
    'gold': 350,
    'silver': 430
}

PISTON_REGISTERS = {
    'blue': 16389,
    'red': 16390,
    'green': 16391,
    'gold': 16392,
    'silver': 16393
}

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

GRACE_FRAMES = 0
ENCODER_REGISTER = 24575
SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255

period_reg = 16395
period_val = 200
#period_val = 0

encoder_reg = 16394
encoder_value = 0

SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 9600

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

        if count > best_count and count > 500:
            best_count = count
            best_color = color

    # Fallback: check if silver is present even if not dominant
    if best_color is None:
        for lower, upper in HSV_RANGES['silver']:
            silver_mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
            silver_count = cv2.countNonZero(silver_mask)
            if silver_count > 500:
                return 'silver'

    return best_color if best_count > 2000 else None

if __name__ == '__main__':
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)
    camera = init_camera()
    grace_counter = 0

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # wait for Arduino to reset
    print("Connected to Arduino")

    if client.connect():
        print("Connected to Modbus server")

        try:
            while camera.IsGrabbing():


                client.write_register(encoder_reg, encoder_value, slave=SLAVE_ID)

                # Write motor speeds
                for name, speed in MOTOR_SPEEDS.items():
                    client.write_register(CONTROL_REGISTERS[name], speed, slave=SLAVE_ID)

                client.write_register(period_reg, period_val + 1, slave=SLAVE_ID)
                client.write_register(period_reg, period_val, slave=SLAVE_ID)

                line = ser.readline().decode('utf-8').strip()
                if line.isdigit():
                    encoder_value = int(line)

                piston = input("Piston to fire: ")
                if piston == '1':
                    client.write_register(PISTON_REGISTERS['blue'], 1, slave=SLAVE_ID)
                    print(f'Fired blue piston at encoder value: {encoder_value}\n')
                    time.sleep(0.05)
                    client.write_register(PISTON_REGISTERS['blue'], 0, slave=SLAVE_ID)
                elif piston == '2':
                    client.write_register(PISTON_REGISTERS['red'], 1, slave=SLAVE_ID)
                    print(f'Fired red piston at encoder value: {encoder_value}\n')
                    time.sleep(0.05)
                    client.write_register(PISTON_REGISTERS['red'], 0, slave=SLAVE_ID)
                elif piston == '3':
                    client.write_register(PISTON_REGISTERS['green'], 1, slave=SLAVE_ID)
                    print(f'Fired green piston at encoder value: {encoder_value}\n')
                    time.sleep(0.05)
                    client.write_register(PISTON_REGISTERS['green'], 0, slave=SLAVE_ID)
                elif piston == '4':
                    client.write_register(PISTON_REGISTERS['gold'], 1, slave=SLAVE_ID)
                    print(f'Fired gold piston at encoder value: {encoder_value}\n')
                    time.sleep(0.05)
                    client.write_register(PISTON_REGISTERS['gold'], 0, slave=SLAVE_ID)
                elif piston == '5':
                    client.write_register(PISTON_REGISTERS['silver'], 1, slave=SLAVE_ID)
                    print(f'Fired silver piston at encoder value: {encoder_value}\n')
                    time.sleep(0.05)
                    client.write_register(PISTON_REGISTERS['silver'], 0, slave=SLAVE_ID)

                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    img = grabResult.Array
                    roi = img[ROI[2]:ROI[3], ROI[0]:ROI[1]]
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    detected_color = None
                    if grace_counter == 0:
                        current_detection = classify_color(hsv)
                        print(current_detection)
                        if current_detection:
                            detected_color = current_detection
                            grace_counter = GRACE_FRAMES

                            target_tick = encoder_value + PISTON_ENCODER_OFFSETS[current_detection]
                            piston_queues[current_detection].append(target_tick)
                            print(f"{index} | Detected: {current_detection} | At encoder value: {encoder_value}")
                            index += 1
                        last_detection = current_detection
                    else:
                        grace_counter -= 1

                    # Overlay text for visual debugging
                    #cv2.putText(roi, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
    




