from pymodbus.client import ModbusTcpClient
import serial
import time
import pandas as pd
import numpy as np
import cv2

SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 9600

SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255

enc_ticks_per_sec = (2048 * (2000 // 20)) // 60

PISTON_REGISTERS = {
    '1': 16389,
    '2': 16390,
    '3': 16391,
    '4': 16392,
    '5': 16393
}

PISTON_START_VALS = {name: 0 for name in PISTON_REGISTERS}

# === Kalman Filter Setup ===

dt = 0.01  # default timestep (will be updated live)

x = np.array([[3503.2], [0.0]])  # Initial state [position, velocity]
P = np.array([[34.2, 0.0], [0.0, 10.0]])  # Initial covariance
H = np.array([[1, 0]])  # We measure position only
R = np.array([[34.2]])  # Measurement noise
Q = np.array([[1e-3, 0], [0, 1e-1]])  # Process noise

def kalman_step(z, x, P, dt):
    F = np.array([[1, dt], [0, 1]])  # update with actual dt
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_new = x_pred + K @ y
    P_new = (np.eye(2) - K @ H) @ P_pred

    return x_new, P_new

# === GUI ===
def init_gui():
    cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
    for name in PISTON_REGISTERS:
        cv2.createTrackbar(f'{name} toggle', 'Control Panel', PISTON_START_VALS[name], 1, lambda x: None)

# === Main ===
def main():
    init_gui()
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    if not ser:
        raise AssertionError("serial failed to connect")
    print("serial connected")

    buffer = b''
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)
    if not client.connect():
        raise AssertionError("failed to connect to modbus")
    print("modbus connected")

    cur_time = time.time()
    cur_encoder = 0
    prev_encoder = 0
    encoder_position = 0  # running total of encoder ticks
    analytics = []
    cur_firing = []
    last_piston_vals = PISTON_START_VALS.copy()

    # === Kalman Filter Initialization ===
    x = np.array([[3503.2], [0.0]])  # Initial state [position, velocity]
    P = np.array([[34.2, 0.0], [0.0, 10.0]])  # Initial covariance
    H = np.array([[1, 0]])  # We measure position only
    R = np.array([[34.2]])  # Measurement noise
    Q = np.array([[1e-3, 0], [0, 1e-1]])  # Process noise

    try:
        while True:
            while ser.in_waiting:
                buffer += ser.read(ser.in_waiting)

                lines = buffer.split(b'\n')

                if lines:
                    line = lines[-2].decode('utf-8').strip()

                    if line.isdigit():
                        cur_encoder = int(line)
                        if prev_encoder == 0:
                            prev_encoder = cur_encoder                  
                    buffer = lines[-1]

            prev_time = cur_time
            cur_time = time.time()
            delta_time = cur_time - prev_time

            for name in PISTON_REGISTERS:
                new_val = cv2.getTrackbarPos(f'{name} toggle', 'Control Panel')
                if new_val != last_piston_vals[name]:
                    cur_firing = []
                    print(cur_firing)
                    client.write_register(PISTON_REGISTERS[name], new_val, slave=SLAVE_ID)
                    if new_val == 1:
                        cur_firing.append(name)
                    last_piston_vals[name] = new_val

            blank = np.zeros((1, 500), dtype=np.uint8)
            cv2.imshow('Control Panel', blank)
            cv2.waitKey(1)

            if delta_time < 1:
                cur_time = prev_time  # optional: maintain prior time if skipping
                continue

            delta_encoder = cur_encoder - prev_encoder
            prev_encoder = cur_encoder
            encoder_position += delta_encoder

            # === Kalman Filter ===
            z = np.array([[encoder_position]])
            x, P = kalman_step(z, x, P, delta_time)

            est_position = float(x[0])
            est_velocity = float(x[1])

            data = {
                'timestamp': cur_time,
                'delta_time': delta_time,
                'delta_encoder': delta_encoder,
                'filtered_position': est_position,
                'filtered_velocity': est_velocity,
                'cur_firing': cur_firing,
                'num_pistons': len(cur_firing)
            }
            analytics.append(data)

            print(f"!!!{delta_encoder}!!!")

    except KeyboardInterrupt:
        print("exiting...")

    finally:
        if len(analytics) > 0:
            pd.DataFrame(analytics).to_csv(f'./enc_analytics/detection_{time.time()}.csv')
        cv2.destroyAllWindows()
        client.close()

if __name__ == '__main__':
    main()
