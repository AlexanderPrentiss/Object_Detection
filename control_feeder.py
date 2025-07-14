from pymodbus.client import ModbusTcpClient
import cv2
import numpy as np



SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255

CONTROL_REGISTERS = {
    'rough1_rough2': 16383,
    'auger': 16385,
    'rough3': 16384,
    'belt1': 16386,
    'belt2': 16387,
}

period_reg = 16388
period_val = 200

MOTOR_SPEEDS = {
    'rough1_rough2': 0,
    'auger': 0,
    'rough3': 0,
    'belt1': 0,
    'belt2': 0,
}

MAX_PWM = 200
LIMITED_MAX_PWM = int(MAX_PWM * 0.5)

def init_gui():
    cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)

    for name in MOTOR_SPEEDS:
        max_val = LIMITED_MAX_PWM if name in ['rough1_rough2', 'rough3'] else MAX_PWM
        cv2.createTrackbar(f'{name}_speed', 'Control Panel', MOTOR_SPEEDS[name], max_val, lambda x: None)

def main():
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)

    if not client.connect():
        print("Couldn't connect to Modbus server")
        return

    init_gui()
    
    print("Connected to Modbus server")

    last_motor_speeds = MOTOR_SPEEDS.copy()

    try:
        while True:
            # === Update GUI-controlled values ===

            # Update motor speeds only if changed
            motors_updated = False
            for name in MOTOR_SPEEDS:
                max_val = LIMITED_MAX_PWM if name in ['rough1_rough2', 'rough3', 'auger'] else MAX_PWM
                new_speed = min(cv2.getTrackbarPos(f'{name}_speed', 'Control Panel'), max_val)

                if new_speed != last_motor_speeds[name]:
                    client.write_register(CONTROL_REGISTERS[name], new_speed, slave=SLAVE_ID)
                    last_motor_speeds[name] = new_speed
                    client.write_register(period_reg, period_val - 1, slave=SLAVE_ID)
                    client.write_register(period_reg, period_val, slave=SLAVE_ID)
            blank = np.zeros((1, 500), dtype=np.uint8)  # 1-pixel tall, wide enough for sliders
            cv2.imshow('Control Panel', blank)
            cv2.waitKey(1)

    finally:
        client.close()
        for name in MOTOR_SPEEDS:
            if name != 'belt2':
                client.write_register(CONTROL_REGISTERS[name], 0, slave=SLAVE_ID)
        client.write_register(period_reg, period_val - 1, slave=SLAVE_ID)
        client.write_register(period_reg, period_val, slave=SLAVE_ID)

if __name__ == '__main__':
    main()


