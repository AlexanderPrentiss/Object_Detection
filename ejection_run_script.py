from pymodbus.client import ModbusTcpClient
from pypylon import pylon
import time
import serial
import cv2
import numpy as np
import pandas as pd

SERVER_IP = '192.168.100.2'
SERVER_PORT = 502
SLAVE_ID = 255

PISTON_REGISTERS = {
    'purple': 16389,
    'red': 16390,
    'blue': 16391,
    'gold': 16392,
    'green': 16393
}

status_reg = 24575

ROI_SLICE = (slice(0, 150), slice(575, 700))
ROI_HEIGHT = ROI_SLICE[0].stop - ROI_SLICE[0].start
ROI_WIDTH = ROI_SLICE[1].stop - ROI_SLICE[1].start

HSV_RANGES = {
    'blue':   [([90, 80, 60], [130, 255, 255])],
    'red':    [([0, 80, 0], [10, 255, 255]), ([160, 80, 1], [179, 255, 255])],
    'green':  [([40, 85, 0], [110, 255, 255])],
    'gold':   [([15, 60, 55], [25, 255, 255])],
    'purple': [([140, 0, 60], [155, 225, 225])],
    'belt':   [([0, 0, 0], [50, 50, 0])],
}

SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 9600

PISTON_ENCODER_OFFSETS = {
    'purple': 715,
    'red': 1370,
    'green': 3410,
    'gold': 2700,
    'blue': 2050
}

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

def classify_color(roi_hsv):
    # Glare suppression mask (from V channel)
    _, _, v = cv2.split(roi_hsv)
    glare_mask = cv2.inRange(v, 254, 255)
    non_glare_mask = cv2.bitwise_not(glare_mask)

    # Belt suppression mask (from belt hsv range)
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
            mask = cv2.inRange(roi_hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            mask_total = cv2.bitwise_or(mask_total, mask)

        mask_total = cv2.bitwise_and(mask_total, non_belt_mask)
        mask_total = cv2.bitwise_and(mask_total, non_glare_mask)

        kernel = np.ones((2, 2), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        count = cv2.countNonZero(mask_total)
        if count > best_count:
            best_color = color
            best_count = count

    return (best_color, best_count) if best_count > 1300 else (None, 0)


MAX_PWM = 200
LIMITED_MAX_PWM = int(MAX_PWM * 0.5)

def init_gui():
    cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)

    for color in PISTON_ENCODER_OFFSETS:
        cv2.createTrackbar(f'{color}_offset', 'Control Panel', PISTON_ENCODER_OFFSETS[color], 5000, lambda x: None)

    for name in MOTOR_SPEEDS:
        max_val = LIMITED_MAX_PWM if name in ['rough1_rough2', 'rough3', 'auger'] else MAX_PWM
        cv2.createTrackbar(f'{name}_speed', 'Control Panel', MOTOR_SPEEDS[name], max_val, lambda x: None)

def main():
    client = ModbusTcpClient(SERVER_IP, port=SERVER_PORT)

    if not client.connect():
        print("Couldn't connect to Modbus server")
        return

    init_gui()
    
    print("Connected to Modbus server")

    camera = init_camera()

    prev_time = time.time()

    prev_detection = None
    detection_count = 0
    prev_voted_color = None

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    buffer = b''
    encoder_value = 0

    start_tick = 0
    go_vote = False
    votes = []

    piston_queues = {color: [] for color in PISTON_REGISTERS.keys()}

    last_motor_speeds = MOTOR_SPEEDS.copy()

    analytics_start_time = time.time()

    detection_analytics = []
    current_detection_analytic = {}

    firing_analytics = []

    warmup_start_time = time.time()

    print('starting serial...(3s)')
    while True:
        buffer_time = time.time()
        buffer += ser.read(ser.in_waiting)
        if buffer_time - warmup_start_time > 3:
            break

    try:
        while True:
            # === Update GUI-controlled values ===

            # Update piston offsets
            for color in PISTON_ENCODER_OFFSETS:
                PISTON_ENCODER_OFFSETS[color] = cv2.getTrackbarPos(f'{color}_offset', 'Control Panel')

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

            camera_grab_result = camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)

            if camera_grab_result.GrabSucceeded():
                frame = camera_grab_result.Array[ROI_SLICE].copy()
            camera_grab_result.Release()

            # Convert to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Generate color masks for each target color
            masks = []
            for color in ['blue', 'red', 'green', 'gold', 'purple']:
                mask_total = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
                for lower, upper in HSV_RANGES[color]:
                    lower_np = np.array(lower, dtype=np.uint8)
                    upper_np = np.array(upper, dtype=np.uint8)
                    mask = cv2.inRange(hsv_frame, lower_np, upper_np)
                    mask_total = cv2.bitwise_or(mask_total, mask)

                # Morphology
                kernel = np.ones((2, 2), np.uint8)
                mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
                mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

                # Add to debug GUI
                mask_color = cv2.cvtColor(mask_total, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask_color, color.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                masks.append(mask_color)

            # Resize all images to uniform size
            resized_frame = cv2.resize(frame, (ROI_WIDTH, ROI_HEIGHT))
            resized_masks = [cv2.resize(m, (ROI_WIDTH, ROI_HEIGHT)) for m in masks]

            # Build 2x3 grid
            row1 = np.hstack([resized_frame, resized_masks[0], resized_masks[1]])  # original, blue, red
            row2 = np.hstack([resized_masks[2], resized_masks[3], resized_masks[4]])  # green, gold, Purple
            debug_grid = np.vstack([row1, row2])

            current_time = time.time()
            delta = current_time - prev_time
            fps = 1.0 / delta if delta > 0 else 0.0
            prev_time = current_time

            while ser.in_waiting:
                buffer += ser.read(ser.in_waiting)
                lines = buffer.split(b'\n')

                if lines:
                    line = lines[-2].decode('utf-8').strip()
                    if line.isdigit():
                        encoder_value = int(line)
                    buffer = lines[-1]

                # Save trailing partial line for next loop
                buffer = lines[-1] if not buffer.endswith(b'\n') else b''

            determined_color = None

            current_detection, pixel_count = classify_color(hsv_frame)

            if current_detection is not None and prev_detection is None:
                current_detection_analytic['start_time'] = time.time() - analytics_start_time
                start_tick = encoder_value
                current_detection_analytic['start_encoder_tick'] = start_tick
                go_vote = True

            elif current_detection is None and prev_detection is not None:
                current_detection_analytic['end_time'] = time.time() - analytics_start_time
                detection_count += 1
                mid_point = (start_tick + encoder_value) // 2
                current_detection_analytic['mid_encoder_tick'] = mid_point
                current_detection_analytic['end_encoder_tick'] = encoder_value
                go_vote = False

                if votes:
                    # Weighted vote logic
                    vote_counts = {}
                    for color, count in votes:
                        vote_counts[color] = vote_counts.get(color, 0) + count

                    determined_color = max(vote_counts, key=vote_counts.get)
                    current_detection_analytic['determined_color'] = determined_color
                    prev_voted_color = determined_color
                    piston_queues[determined_color].append(mid_point + PISTON_ENCODER_OFFSETS[determined_color])
                    print(f'{determined_color} | {piston_queues[determined_color][-1]}\nVote results: {vote_counts}')

                votes = []
                detection_analytics.append(current_detection_analytic)
                current_detection_analytic = {}

            prev_detection = current_detection

            if go_vote and current_detection:
                votes.append((current_detection, pixel_count))

            for color, queue in piston_queues.items():
                if queue and encoder_value > queue[0]:
                    current_firing_analytic = {
                        'time': time.time() - analytics_start_time,
                        'tried': True,
                        'color': color,
                        'encoder_value': encoder_value,
                        'target_encoder_value': queue[0],
                    }
                    client.write_register(PISTON_REGISTERS[color], 1, slave=SLAVE_ID)
                    time.sleep(0.01)
                    client.write_register(PISTON_REGISTERS[color], 0, slave=SLAVE_ID)


                    current_firing_analytic['fired'] = client.read_input_registers(status_reg, count=1, slave=SLAVE_ID).status

                    queue.pop(0)
                    firing_analytics.append(current_firing_analytic)

            cv2.putText(
                debug_grid,
                f'FPS: {fps:.1f}',
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                debug_grid,
                f'Enc: {encoder_value}',
                (0, ROI_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            top_right_y = 30
            line_spacing = 30

            cv2.putText(
                debug_grid,
                f'Detections: {detection_count}',
                (ROI_WIDTH - 250, top_right_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                debug_grid,
                f'Current Color: {current_detection}',
                (ROI_WIDTH - 250, top_right_y + line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                debug_grid,
                f'Voted Color: {prev_voted_color}',
                (ROI_WIDTH - 250, top_right_y + 2*line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.imshow('Debug Grid', debug_grid)
            cv2.waitKey(1)
            
    finally:
        if len(detection_analytics) > 0:
            pd.DataFrame(detection_analytics).to_csv(f'./analytics/detection_{time.time()}.csv')
        if len(firing_analytics) > 0:
            pd.DataFrame(firing_analytics).to_csv(f'./analytics/firing_{time.time()}.csv')
        client.close()
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()