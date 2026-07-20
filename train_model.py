import cv2
import os
import numpy as np
import mysql.connector
import face_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_FILE = os.path.join(BASE_DIR, "trainer/trainer.yml")

# Connect to database to fetch active registered IDs
try:
    import config
    conn = config.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM staff")
    valid_ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    print(f"Database synced. Active IDs in DB: {valid_ids}")
except Exception as e:
    print(f"Warning: Could not connect to database ({e}). Training on all folders.")
    valid_ids = None

import pickle

recognizer = face_utils.get_sface_recognizer()

database_embeddings = {}
total_images = 0

print("Reading dataset...")

for folder_name in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # folder name format is "<id>_<name>", e.g. "3_rahul"
    try:
        person_id = int(folder_name.split("_")[0])
        if valid_ids is not None and person_id not in valid_ids:
            print(f"Skipping folder (not registered in database): {folder_name}")
            continue
    except ValueError:
        print(f"Skipping folder with unexpected name: {folder_name}")
        continue

    person_embeddings = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        # SFace expects 112x112 aligned color images.
        # If shape is 112x112, extract embedding directly.
        if img.shape[0] == 112 and img.shape[1] == 112:
            emb = recognizer.feature(img)
        else:
            # Fallback for legacy images
            detector = face_utils.get_yunet_detector(img.shape[1], img.shape[0])
            retval, faces = detector.detect(img)
            if faces is not None and len(faces) > 0:
                aligned = recognizer.alignCrop(img, faces[0])
                emb = recognizer.feature(aligned)
            else:
                continue

        if emb is not None:
            person_embeddings.append(emb)
            total_images += 1

    if person_embeddings:
        database_embeddings[person_id] = person_embeddings
        print(f"  {folder_name}: {len(person_embeddings)} embeddings extracted")

if total_images == 0:
    print("No valid face images found in dataset/. Please register at least one person first.")
else:
    print(f"\nExtracted {total_images} embeddings from {len(database_embeddings)} people...")
    
    trainer_dir = os.path.dirname(TRAINER_FILE)
    os.makedirs(trainer_dir, exist_ok=True)
    pickle_path = os.path.join(trainer_dir, "embeddings.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(database_embeddings, f)
    print(f"Training complete. Embeddings saved to {pickle_path}")
