from pymodbus.client import ModbusTcpClient
import time

SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255

PISTON_REGISTERS = {
    'purple': 16389,
    'red': 16390,
    'green': 16391,
    'gold': 16392,
    'blue': 16393
}

def fire_pistons(client, pistons):
    if len(pistons) == 0:
        return
    for name in pistons:
                client.write_register(PISTON_REGISTERS[name], 1, slave=SLAVE_ID)
                time.sleep(0.03)
                client.write_register(PISTON_REGISTERS[name], 0, slave=SLAVE_ID)    

def main():
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)

    if not client.connect():
        raise AssertionError("failed to connect to modbus")
    
    print("connected to modbus")

    try:
        while True:
            fire_pistons(client, PISTON_REGISTERS)
            time.sleep(1)
    except KeyboardInterrupt:
        print("exiting...")
    finally:
        for name in PISTON_REGISTERS:
            client.write_register(PISTON_REGISTERS[name], 0, slave=SLAVE_ID)

if __name__ == '__main__':
    main()
