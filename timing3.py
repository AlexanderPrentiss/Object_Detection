import serial
import cv2
import numpy as np
from pymodbus.client import ModbusTcpClient
import time
from pypylon import pylon
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading

# HSV range for the belt color
HSV_RANGES = {
    'belt': [
        ([0, 0, 0], [180, 255, 100])
    ]
}

# Modbus + serial config
piston_reg = 16383
encoder_reg = 16388
SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255
SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 9600

# Graphing data
encoder_value = 0
start_time = time.time()
times = deque(maxlen=1000)
values = deque(maxlen=1000)

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
    return object_pixels > 1000

def update_plot(frame):
    ax.clear()
    ax.plot(times, values, label='Encoder')
    ax.set_title("Encoder Value Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Encoder Count")
    ax.grid(True)

    # Sliding 30-second window
    if times:
        xmax = times[-1]
        xmin = max(0, xmax - 30)
        ax.set_xlim(xmin, xmax)

def run_main_loop():
    global encoder_value

    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    camera = init_camera()

    detecting = False
    entry_position = 0
    exit_position = 0

    if client.connect():
        try:
            while True:
                # Read encoder
                line = ser.readline().decode('utf-8').strip()
                if line.isdigit():
                    encoder_value = int(line)
                    now = time.time() - start_time
                    times.append(now)
                    values.append(encoder_value)
                    client.write_register(encoder_reg, encoder_value & 0xFFFF, slave=SLAVE_ID)
                    print(f'Encoder Value: {encoder_value}')

                # Grab and process image
                if camera.IsGrabbing():
                    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                    if grab_result.GrabSucceeded():
                        frame = grab_result.Array
                        roi = frame[300:400, 300:600]
                        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                        object_present = detect_object_presence(roi_hsv)

                        if object_present and not detecting:
                            detecting = True
                            entry_position = encoder_value

                        elif not object_present and detecting:
                            detecting = False
                            exit_position = encoder_value
                            midpoint = (entry_position + exit_position) // 2
                            safe_value = min(max(midpoint + 200, 0), 65535)
                            client.write_register(piston_reg, safe_value, slave=SLAVE_ID)
                            print(f"Object detected from {entry_position} to {exit_position}. Midpoint: {midpoint}")

                    grab_result.Release()
        except KeyboardInterrupt:
            print("Exiting main loop.")
        finally:
            client.close()
            camera.StopGrabbing()

# Run data loop in background
main_thread = threading.Thread(target=run_main_loop)
main_thread.daemon = True
main_thread.start()

# Setup and show the plot
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update_plot, interval=100)
plt.tight_layout()
plt.show()

