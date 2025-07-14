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

PISTON_REGISTERS = {
    '1': 16389,
    '2': 16390,
    '3': 16391,
    '4': 16392,
    '5': 16393
}

PISTON_START_VALS = {
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0
}

def init_gui():
    cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)

    for name in PISTON_REGISTERS:
        cv2.createTrackbar(f'{name} toggle', 'Control Panel', PISTON_START_VALS[name], 1, lambda x: None)

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
    cur_encoder = 1
    prev_encoder = 0

    analytics = []

    pistons = []

    cur_firing = []
    new_val = 0

    last_piston_vals = PISTON_START_VALS.copy()

    warmup_start_time = time.time()

    print('starting serial...(3s)')
    while True:
        buffer_time = time.time()
        buffer += ser.read(ser.in_waiting)
        if buffer_time - warmup_start_time > 3:
            break

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
                    if new_val == 1:
                        cur_firing.append(name)
                    last_piston_vals[name] = new_val
            
            blank = np.zeros((1, 500), dtype=np.uint8)
            cv2.imshow('Control Panel', blank)
            cv2.waitKey(1)

            print(f'!!!!!!!{cur_encoder}!!!!!!!')

            if delta_time < 1:
                cur_time = prev_time
                continue

            delta_encoder = cur_encoder - prev_encoder

            prev_encoder = cur_encoder

            data = {
                'delta_time': delta_time,
                'delta_encoder': delta_encoder
            }

            data['cur_firing'] = cur_firing
            data['num_pistons'] = len(cur_firing)

            analytics.append(data)

            for name in cur_firing:
                client.write_register(PISTON_REGISTERS[name], 1, slave=SLAVE_ID)
                time.sleep(0.05)
                client.write_register(PISTON_REGISTERS[name], 0, slave=SLAVE_ID)

    except KeyboardInterrupt:
        print("exiting...")
    finally:
        if len(analytics) > 0:
            pd.DataFrame(analytics).to_csv(f'./enc_analytics/detection_{time.time()}.csv')
        for name in PISTON_REGISTERS:
            client.write_register(PISTON_REGISTERS[name], 0, slave=SLAVE_ID)
        cv2.destroyAllWindows()
        client.close()

if __name__ == '__main__':
    main()




    