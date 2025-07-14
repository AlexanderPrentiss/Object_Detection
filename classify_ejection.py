import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
import serial
from pypylon import pylon
from model import BeltAutoencoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

# ========== Config ==========
MODEL_PATH = "./models/best_model.pt"
DATA_DIR = "./color_classifier_data"
RESIZE_SHAPE = (128, 64)
POST_RESIZE_PAD_WIDTH = 6
ROI_SLICE = (slice(0, 1024), slice(391, 858))
THRESHOLD = 0.10
K = 15
ENCODER_TICKS_PER_ROI = 550  # define this based on your system
ROI_HEIGHT = RESIZE_SHAPE[0]

SERIAL_PORT = '/dev/ttyTHS1'
BAUD_RATE = 9600

# ========== Dummy Encoder Function (replace with real one) ==========
def get_encoder_value(ser):
    while ser.in_waiting:
                buffer += ser.read(ser.in_waiting)
                lines = buffer.split(b'\n')

                if lines:
                    line = lines[-2].decode('utf-8').strip()
                    if line.isdigit():
                        encoder_val = int(line)
                    buffer = lines[-1]

                # Save trailing partial line for next loop
                buffer = lines[-1] if not buffer.endswith(b'\n') else b''
    return encoder_val  # dummy example

# ========== Feature Extraction ==========
def extract_color_features(pixels_rgb):
    pixels_lab = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    pixels_lab = pixels_lab.astype(np.float32) / 255.0
    mean = pixels_lab.mean(axis=0)
    cov = np.cov(pixels_lab, rowvar=False)
    hist, _ = np.histogramdd(pixels_lab, bins=(5, 5, 5), range=[[0, 1], [0, 1], [0, 1]])
    hist = hist.flatten() / hist.sum()
    return np.concatenate([mean, cov.flatten(), hist])

# ========== Load Training Data ==========
def load_dataset(data_dir):
    features = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if file.endswith(".npz"):
                data = np.load(os.path.join(label_path, file))
                feat = np.concatenate([data["mean"], data["cov"].flatten(), data["hist"]])
                features.append(feat)
                labels.append(label)
    return np.array(features), np.array(labels)

print(":mag: Loading dataset...")
X, y = load_dataset(DATA_DIR)
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_scaled, y)
print(":white_check_mark: KNN model trained.")

# ========== Load Autoencoder ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeltAutoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ========== Init Camera ==========
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
resize = cv2.resize

# Color mapping
CLASS_COLORS = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "gold": (0, 215, 255),
    "purple": (255, 0, 255),
    "belt": None  # Don't draw
}

# Overlay FPS
def add_info_bar(image, text, height=30, font_scale=0.5, thickness=1):
    bar = np.zeros((height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, int(height * 0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([bar, image])

# Residual
def compute_residual(input_img, recon_img):
    return np.linalg.norm(input_img - recon_img, axis=-1)

# ========== Tracking Variables ==========
object_id_counter = 0
tracked_objects = {}  # id: centroid (x, y)
object_history = {}   # id: list of (y, encoder_bottom)
object_labels = {}    # id: classification label

# ========== Main Loop ==========
print(":movie_camera: Running detection and classification. Press 'q' to quit.")
prev_time = time.time()

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

print('starting serial...(3s)')
while True:
    buffer_time = time.time()
    buffer += ser.read(ser.in_waiting)
    if buffer_time - prev_time > 3:
        break

while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    frame = grab_result.Array[ROI_SLICE]
    grab_result.Release()

    frame_resized = resize(frame, (RESIZE_SHAPE[1] - POST_RESIZE_PAD_WIDTH, RESIZE_SHAPE[0]), interpolation=cv2.INTER_AREA)
    frame_tensor = torch.tensor(frame_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    frame_tensor = F.pad(frame_tensor, (0, POST_RESIZE_PAD_WIDTH, 0, 0), value=0).unsqueeze(0).to(device)

    with torch.no_grad():
        recon_tensor = model(frame_tensor)

    input_img = frame_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    recon_img = recon_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    residual = compute_residual(input_img, recon_img)

    input_bgr = (input_img * 255).astype(np.uint8)
    recon_bgr = (recon_img * 255).astype(np.uint8)
    residual_colormap = cv2.applyColorMap((residual * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    binary_mask = (residual > THRESHOLD).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    encoder_bottom = get_encoder_value(ser)
    rects = [cv2.boundingRect(c) for c in contours]
    centroids = np.array([((x + x + w) // 2, (y + y + h) // 2) for (x, y, w, h) in rects])

    new_tracked = {}
    if len(tracked_objects) == 0:
        for centroid in centroids:
            tracked_objects[object_id_counter] = centroid
            object_history[object_id_counter] = [(centroid[1], encoder_bottom)]
            object_id_counter += 1
    else:
        old_ids = list(tracked_objects.keys())
        old_centroids = np.array([tracked_objects[i] for i in old_ids])
        if len(old_centroids) > 0 and len(centroids) > 0:
            D = cdist(old_centroids, centroids)
            row_idxs = D.min(axis=1).argsort()
            col_idxs = D.argmin(axis=1)[row_idxs]
            used_cols = set()
            for row, col in zip(row_idxs, col_idxs):
                if col in used_cols:
                    continue
                obj_id = old_ids[row]
                centroid = centroids[col]
                new_tracked[obj_id] = centroid
                object_history[obj_id].append((centroid[1], encoder_bottom))
                used_cols.add(col)
            for i, centroid in enumerate(centroids):
                if i not in used_cols:
                    new_tracked[object_id_counter] = centroid
                    object_history[object_id_counter] = [(centroid[1], encoder_bottom)]
                    object_id_counter += 1
    tracked_objects = new_tracked

    # Classification + Annotation
    CONFIDENCE_THRESHOLD = 0.75
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        pixels = input_bgr[mask == 255]
        if pixels.size == 0:
            continue

        feat = extract_color_features(pixels)
        feat_scaled = scaler.transform([feat])
        proba = knn.predict_proba(feat_scaled)[0]
        pred_idx = np.argmax(proba)
        pred_label = knn.classes_[pred_idx]
        confidence = proba[pred_idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            color = CLASS_COLORS.get(pred_label)
            if color is not None:
                cv2.drawContours(input_bgr, [contour], -1, color, 2)

    # Final position estimation
    done_ids = []
    for obj_id, (cx, cy) in tracked_objects.items():
        if cy > ROI_HEIGHT:
            done_ids.append(obj_id)

    for obj_id in done_ids:
        encoder_estimates = [
            encoder_bottom - (1 - y / ROI_HEIGHT) * ENCODER_TICKS_PER_ROI
            for y, encoder_bottom in object_history[obj_id]
        ]
        avg_encoder = np.mean(encoder_estimates)
        print(f"Object ID {obj_id} final encoder position: {avg_encoder:.2f}")
        del tracked_objects[obj_id]
        del object_history[obj_id]

    # Visualize tracked objects
    for obj_id, (cx, cy) in tracked_objects.items():
        cv2.circle(input_bgr, (cx, cy), 4, (255, 255, 255), -1)
        cv2.putText(input_bgr, f"ID {obj_id}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # === Visualization ===
    vis_resize = (512, 1024)
    vis_input = resize(input_bgr, vis_resize)
    vis_recon = resize(recon_bgr, vis_resize)
    vis_resid = resize(residual_colormap, vis_resize)
    combined = np.hstack((vis_input, vis_recon, vis_resid))
    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time)
    prev_time = cur_time
    combined = add_info_bar(combined, f"FPS: {fps:.2f}")
    cv2.imshow("Detected | Reconstruction | Residual", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========== Cleanup ==========
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()