import serial
import cv2
import numpy as np
from pymodbus.client import ModbusTcpClient
import time
import matplotlib as plt

HSV_RANGES = {
    'belt': [
        ([0, 0, 0], [180, 255, 60])  # Very dark pixels (black plastic bristle belt)
    ],
    'red': [
        ([0, 100, 100], [10, 255, 255]),
        ([160, 100, 100], [179, 255, 255])
    ],
    'blue': [
        ([100, 120, 70], [130, 255, 255])
    ],
    'green': [
        ([40, 100, 40], [85, 255, 255])
    ],
    'gold': [
        ([15, 100, 120], [35, 255, 255])
    ],
    'silver': [
        ([0, 0, 160], [179, 40, 255])  # Bright and desaturated
    ]
}

piston_reg = 16383
encoder_reg = 16388

SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255

SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 9600

encoder_value = 0

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

def detect_object_presence(roi_hsv):
    # Smooth the image to reduce noise
    roi_hsv = cv2.GaussianBlur(roi_hsv, (5, 5), 0)

    belt_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES['belt']:
        mask = cv2.inRange(roi_hsv, np.array(lower), np.array(upper))
        belt_mask = cv2.bitwise_or(belt_mask, mask)

    object_mask = cv2.bitwise_not(belt_mask)
    object_pixels = cv2.countNonZero(object_mask)

    return object_pixels > 1000  # Threshold may be tuned for your ROI size

if __name__ == '__main__':
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    camera = init_camera()

    if client.connect():
        try:
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line.isdigit():
                    encoder_value = int(line)

                client.write_register(encoder_reg, encoder_value, slave=SLAVE_ID)
                print(f'Encoder Value: {encoder_value}.\n')
        except KeyboardInterrupt:
            print("exiting...")
        finally:
            client.close()
