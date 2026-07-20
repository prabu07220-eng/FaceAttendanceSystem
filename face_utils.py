import os
import requests
import cv2
import numpy as np

# Path definitions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
LANDMARKER_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
YOLO_ONNX_PATH = os.path.join(MODELS_DIR, "yolov5n.onnx")

YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
YOLO_ONNX_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunk
                    if chunk:
                        f.write(chunk)
            print("Download complete.")
        else:
            raise Exception(f"Failed to download. HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # If download fails, try using curl as a backup (often LFS works better on Windows system curl)
        import subprocess
        print("Trying fallback download via curl...")
        try:
            subprocess.run(["curl", "-L", "-o", target_path, url], check=True)
            print("Curl fallback download complete.")
        except Exception as curl_err:
            print(f"Curl fallback failed: {curl_err}")
            raise e

def download_models_if_needed():
    if not os.path.exists(YUNET_PATH):
        download_file(YUNET_URL, YUNET_PATH)
    if not os.path.exists(SFACE_PATH):
        download_file(SFACE_URL, SFACE_PATH)
    if not os.path.exists(LANDMARKER_PATH):
        download_file(LANDMARKER_URL, LANDMARKER_PATH)
    if not os.path.exists(YOLO_ONNX_PATH):
        download_file(YOLO_ONNX_URL, YOLO_ONNX_PATH)

def get_yunet_detector(width, height):
    download_models_if_needed()
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_PATH,
        config="",
        input_size=(width, height),
        score_threshold=0.8,  # Slightly lower to capture side views / moving faces
        nms_threshold=0.3,
        top_k=5000
    )
    return detector

def get_sface_recognizer():
    download_models_if_needed()
    recognizer = cv2.FaceRecognizerSF.create(
        model=SFACE_PATH,
        config=""
    )
    return recognizer

def extract_embedding(img, face_row, recognizer):
    """
    Aligns the face and extracts the 128-dimensional embedding vector.
    """
    if img is None or face_row is None:
        return None
    aligned = recognizer.alignCrop(img, face_row)
    embedding = recognizer.feature(aligned)
    return embedding

def compare_embeddings(emb1, emb2, recognizer):
    """
    Compares two embeddings using Cosine Similarity.
    SFace Cosine Similarity: Match threshold is typically 0.363.
    """
    if emb1 is None or emb2 is None:
        return 0.0
    return recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)


# --- Liveness Detection Helpers ---
import sys
from types import ModuleType
if "matplotlib" not in sys.modules:
    sys.modules["matplotlib"] = ModuleType("matplotlib")
if "matplotlib.pyplot" not in sys.modules:
    sys.modules["matplotlib.pyplot"] = ModuleType("pyplot")

import mediapipe as mp

face_landmarker_instance = None

def get_face_mesh_landmarks(face_crop):
    """
    Runs MediaPipe Face Mesh on the cropped face ROI and returns 3D landmarks relative to crop.
    """
    global face_landmarker_instance
    if face_crop is None or face_crop.size == 0:
        return None
    try:
        if face_landmarker_instance is None:
            download_models_if_needed()
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            face_landmarker_instance = vision.FaceLandmarker.create_from_options(options)

        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        results = face_landmarker_instance.detect(mp_image)
        if results.face_landmarks:
            landmarks = []
            for lm in results.face_landmarks[0]:
                landmarks.append([lm.x, lm.y, lm.z])
            return np.array(landmarks)
    except Exception as e:
        print(f"Error extracting face landmarks: {e}")
    return None

def calculate_ear(landmarks):
    """
    Calculates Eye Aspect Ratio (EAR) using standard MediaPipe indices.
    """
    if landmarks is None or len(landmarks) < 468:
        return 0.3 # Default open eye value
    try:
        # Left eye: outer=33, top=160, 158, inner=133, bottom=153, 144
        l_v1 = np.linalg.norm(landmarks[160] - landmarks[144])
        l_v2 = np.linalg.norm(landmarks[158] - landmarks[153])
        l_h  = np.linalg.norm(landmarks[33] - landmarks[133])
        left_ear = (l_v1 + l_v2) / (2.0 * l_h) if l_h > 0 else 0.0

        # Right eye: outer=263, top=385, 387, inner=362, bottom=373, 380
        r_v1 = np.linalg.norm(landmarks[385] - landmarks[380])
        r_v2 = np.linalg.norm(landmarks[387] - landmarks[373])
        r_h  = np.linalg.norm(landmarks[263] - landmarks[362])
        right_ear = (r_v1 + r_v2) / (2.0 * r_h) if r_h > 0 else 0.0

        return (left_ear + right_ear) / 2.0
    except Exception:
        return 0.3


# --- YOLOv5 ONNX Mobile Phone & Screen Detector ---
yolo_net = None

def get_yolo_net():
    global yolo_net
    if yolo_net is None:
        download_models_if_needed()
        yolo_net = cv2.dnn.readNet(YOLO_ONNX_PATH)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return yolo_net

def detect_screens(frame):
    """
    Returns a list of screen bounding boxes (x, y, w, h, class_id, confidence)
    for detected screens (tv=62, laptop=63, cell phone=67) in the frame with confidence > 0.20.
    """
    screens = []
    if frame is None or frame.size == 0:
        return screens
    try:
        net = get_yolo_net()
        h_img, w_img = frame.shape[:2]
        # YOLOv5 ONNX requires exactly 640x640 input shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward() # shape: (1, 25200, 85)
        
        detections = output[0]
        for detection in detections:
            box_conf = detection[4]
            if box_conf > 0.20:
                classes_scores = detection[5:]
                class_id = np.argmax(classes_scores)
                confidence = classes_scores[class_id]
                
                # Check for tv (62), laptop (63), cell phone (67)
                if confidence > 0.20 and class_id in [62, 63, 67]:
                    cx, cy, w, h = detection[0:4]
                    x = int((cx - w/2) * w_img)
                    y = int((cy - h/2) * h_img)
                    w = int(w * w_img)
                    h = int(h * h_img)
                    screens.append((x, y, w, h, class_id, confidence))
    except Exception as e:
        print(f"Error in screen detection: {e}")
    return screens


