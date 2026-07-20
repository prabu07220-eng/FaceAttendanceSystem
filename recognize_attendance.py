import cv2
import os
import datetime
import mysql.connector
import openpyxl
import face_utils
import pickle
import numpy as np
import collections

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINER_FILE = os.path.join(BASE_DIR, "trainer/embeddings.pkl")
CONFIDENCE_THRESHOLD = 0.50   # SFace Cosine Similarity: HIGHER = better match. Restored to 0.50 for high precision.
UNKNOWN_DIR = os.path.join(BASE_DIR, "static/unknown_faces")
SPOOF_DIR = os.path.join(BASE_DIR, "static/spoof_attempts")
WINDOW_SIZE = 10            # number of recent frames to look back at
MATCH_RATIO_NEEDED = 0.5    # 50% of recent frames must agree before we trust the result
DEBUG_PRINT = True          # prints raw confidence values to the terminal

os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(SPOOF_DIR, exist_ok=True)


import config

# MySQL connection
conn = config.get_db_connection()
cursor = conn.cursor()

# Load trained embeddings
if not os.path.exists(TRAINER_FILE):
    print("No trained embeddings found. Please run train_model.py first.")
    exit()

database_embeddings = {}
last_file_mtime = os.path.getmtime(TRAINER_FILE)
with open(TRAINER_FILE, "rb") as f:
    database_embeddings = pickle.load(f)
print(f"Loaded {len(database_embeddings)} registered people from embeddings.pkl")

# Initialize models
# We start the camera first to get width and height to configure YuNet
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, test_frame = cam.read()
if not ret:
    print("Could not read from camera.")
    cam.release()
    exit()

height, width = test_frame.shape[:2]
detector = face_utils.get_yunet_detector(width, height)
recognizer = face_utils.get_sface_recognizer()

# Load staff/student details from database into memory: {id: (name, type, department)}
cursor.execute("SELECT id, name, person_type, department FROM staff")
people = {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# Keep track of who has already been marked in this session to avoid duplicate DB hits
last_marked = {}
MARK_COOLDOWN = 10  # seconds between updates for the same person

# Rolling history of recent predictions
prediction_history = {}
last_stable_result = {}   # bucket -> last confirmed (non-"verifying") result, used for hysteresis

last_reload_time = 0

def check_and_reload_embeddings():
    global database_embeddings, last_file_mtime, last_reload_time
    now_ts = datetime.datetime.now().timestamp()
    if now_ts - last_reload_time > 5.0:  # check every 5 seconds
        last_reload_time = now_ts
        if os.path.exists(TRAINER_FILE):
            try:
                mtime = os.path.getmtime(TRAINER_FILE)
                if mtime != last_file_mtime:
                    with open(TRAINER_FILE, "rb") as f_in:
                        database_embeddings = pickle.load(f_in)
                    last_file_mtime = mtime
                    print(f"Reloaded {len(database_embeddings)} people's embeddings from embeddings.pkl")
                    # Also reload database names cache
                    cursor.execute("SELECT id, name, person_type, department FROM staff")
                    people.clear()
                    people.update({row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()})
            except Exception as e:
                print("Error reloading embeddings:", e)

def get_position_bucket(x, y):
    return (x // 120, y // 120)

def calculate_overlap_ratio(box1, box2):
    # box format: (x, y, w, h)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection_area = (x2 - x1) * (y2 - y1)
    box1_area = box1[2] * box1[3]
    if box1_area == 0:
        return 0.0
    return intersection_area / box1_area

def get_stable_prediction(bucket, raw_result):
    if bucket not in prediction_history:
        prediction_history[bucket] = collections.deque(maxlen=WINDOW_SIZE)
    history = prediction_history[bucket]
    history.append(raw_result)

    if len(history) < 3:
        return "verifying"

    counts = collections.Counter(history)
    most_common_result, count = counts.most_common(1)[0]
    ratio = count / len(history)

    previous = last_stable_result.get(bucket)

    if previous is not None and previous != "unknown" and most_common_result == "unknown" and ratio < 0.85:
        return previous

    if ratio >= MATCH_RATIO_NEEDED:
        last_stable_result[bucket] = most_common_result
        return most_common_result

    return previous if previous is not None else "verifying"

def mark_attendance(person_id):
    now = datetime.datetime.now()
    today = now.date()

    cursor.execute(
        "SELECT id, entry_time, exit_time FROM attendance WHERE staff_id=%s AND DATE(entry_time)=%s",
        (person_id, today)
    )
    row = cursor.fetchone()

    if row is None:
        cursor.execute(
            "INSERT INTO attendance (staff_id, entry_time, exit_time) VALUES (%s, %s, %s)",
            (person_id, now, now)
        )
    else:
        cursor.execute(
            "UPDATE attendance SET exit_time=%s WHERE id=%s",
            (now, row[0])
        )
    conn.commit()

def log_unknown(face_img):
    now = datetime.datetime.now()
    filename = f"{UNKNOWN_DIR}/unknown_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, face_img)
    cursor.execute(
        "INSERT INTO unknown_alerts (detected_time, image_path) VALUES (%s, %s)",
        (now, filename)
    )
    conn.commit()
    print(f"Unknown face detected and logged: {filename}")


liveness_history = {}

def log_spoof(face_img):
    now = datetime.datetime.now()
    filename = f"{SPOOF_DIR}/spoof_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, face_img)
    try:
        temp_conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="face_attendance_db2"
        )
        temp_cursor = temp_conn.cursor()
        db_path = f"static/spoof_attempts/spoof_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        temp_cursor.execute(
            "INSERT INTO spoof_alerts (detected_time, image_path) VALUES (%s, %s)",
            (now, db_path)
        )
        temp_conn.commit()
        temp_conn.close()
        print(f"[Anti-Spoof] Spoof attempt logged to DB: {db_path}")
    except Exception as e:
        print(f"Error logging spoof to DB: {e}")


def check_liveness_state(bucket, landmarks, face_crop, raw_result):
    if bucket not in liveness_history:
        liveness_history[bucket] = {
            "blink_detected": False,
            "movement_detected": False,
            "liveness_passed": False,
            "ear_history": collections.deque(maxlen=30),
            "landmarks_history": collections.deque(maxlen=15),
            "consecutive_frames": 0,
            "is_blink_state": False,
            "spoof_alert_logged": False
        }
        
    state = liveness_history[bucket]
    
    if state["liveness_passed"]:
        return "passed"
        
    # Eye Gaze Check: Verify iris center relative to eye corners
    is_looking_at_camera = True
    if len(landmarks) >= 478:
        # Left eye horizontal bounds: outer=33, inner=133. Left pupil=468.
        denom_l = landmarks[133][0] - landmarks[33][0]
        # Right eye horizontal bounds: inner=362, outer=263. Right pupil=473.
        denom_r = landmarks[263][0] - landmarks[362][0]
        
        if abs(denom_l) > 1e-5 and abs(denom_r) > 1e-5:
            gaze_left = (landmarks[468][0] - landmarks[33][0]) / denom_l
            gaze_right = (landmarks[473][0] - landmarks[362][0]) / denom_r
            
            # Direct gaze requires pupils within flexible eye bounds [0.25, 0.75]
            if not (0.25 <= gaze_left <= 0.75 and 0.25 <= gaze_right <= 0.75):
                is_looking_at_camera = False

    if not is_looking_at_camera:
        return "look_at_camera"

    state["consecutive_frames"] += 1
    
    # Calculate EAR
    ear = face_utils.calculate_ear(landmarks)
    state["ear_history"].append(ear)
    
    # Calculate micro-movement
    state["landmarks_history"].append(landmarks)
    
    # Blink Detection logic:
    if ear < 0.22:
        state["is_blink_state"] = True
    elif ear >= 0.22 and state["is_blink_state"]:
        state["blink_detected"] = True
        state["is_blink_state"] = False
        print(f"[Liveness] Blink detected for bucket {bucket}!")
        
    # Micro-movement detection:
    if len(state["landmarks_history"]) >= 8:
        coords = np.array(list(state["landmarks_history"]))
        normalized = coords - coords[:, [4], :]
        std_normalized = np.std(normalized, axis=0)
        mean_std = np.mean(std_normalized)
        
        if mean_std > 0.0002:
            state["movement_detected"] = True
            
    # Liveness check passed conditions (Ultra-fast UX: blink instant, 10 frames movement, 20 frames timeout):
    if state["blink_detected"] or (state["movement_detected"] and state["consecutive_frames"] > 10) or state["consecutive_frames"] > 20:
        state["liveness_passed"] = True
        print(f"[Liveness] Liveness PASSED for bucket {bucket}!")
        return "passed"
        
    # Check if we should flag as spoof attempt:
    if state["consecutive_frames"] > 80:
        if not state["spoof_alert_logged"] and raw_result != "unknown":
            log_spoof(face_crop)
            state["spoof_alert_logged"] = True
        return "spoof"
        
    return "verifying"

def export_attendance_to_excel():
    cursor.execute("""
        SELECT s.id_number, s.name, s.person_type, s.department, a.entry_time, a.exit_time
        FROM attendance a
        JOIN staff s ON a.staff_id = s.id
        ORDER BY a.entry_time DESC
    """)
    rows = cursor.fetchall()

    wb = openpyxl.Workbook()
    headers = ["ID Number", "Name", "Type", "Department", "Status", "Entry Time", "Exit Time"]

    # 1. Students Sheet
    ws_students = wb.active
    ws_students.title = "Students"
    ws_students.append(headers)

    # 2. Staff Sheet
    ws_staff = wb.create_sheet(title="Staff")
    ws_staff.append(headers)

    for row in rows:
        id_number, name, person_type, department, entry_time, exit_time = row
        entry_str = entry_time.strftime("%Y-%m-%d %H:%M:%S") if entry_time else ""
        exit_str = exit_time.strftime("%Y-%m-%d %H:%M:%S") if exit_time else ""
        record = [id_number, name, person_type, department or "—", "Present", entry_str, exit_str]

        if person_type == "Student":
            ws_students.append(record)
        else:
            ws_staff.append(record)

    # Auto-adjust column widths for Students
    for col_cells in ws_students.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col_cells)
        ws_students.column_dimensions[col_cells[0].column_letter].width = max_len + 3

    # Auto-adjust column widths for Staff
    for col_cells in ws_staff.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col_cells)
        ws_staff.column_dimensions[col_cells[0].column_letter].width = max_len + 3

    filename = os.path.join(BASE_DIR, "attendance_report.xlsx")
    try:
        wb.save(filename)
        print(f"\nAttendance report exported to {filename}")
    except PermissionError:
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_filename = os.path.join(BASE_DIR, f"attendance_report_{now_str}.xlsx")
        wb.save(alt_filename)
        print(f"\nWarning: attendance_report.xlsx was locked. Saved as {alt_filename}")

# ---------------- Live recognition loop ----------------

print("Starting live recognition. Press 'q' to stop and export the attendance report.")

frame_count = 0
detected_screens = []

while True:
    ret, frame = cam.read()
    if not ret:
        print("Could not read from camera.")
        break

    frame_count += 1
    if frame_count % 3 == 0 or frame_count == 1:
        detected_screens = face_utils.detect_screens(frame)

    # Dynamic reload check
    check_and_reload_embeddings()

    # Set input size for detector
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    retval, faces = detector.detect(frame)

    active_buckets_this_frame = set()
    face_processed = False

    if faces is not None and len(faces) > 0:
        for face in faces:
            # Bounding box
            x, y, w, h = map(int, face[0:4])
            
            # Ignore small/distant faces in background to prevent noisy false positives
            if w < 60 or h < 60:
                continue
                
            face_processed = True
                
            x, y = max(0, x), max(0, y)
            w, h = min(frame.shape[1] - x, w), min(frame.shape[0] - y, h)
            
            face_crop = frame[y:y + h, x:x + w]
            
            # Align and extract SFace embedding
            aligned = recognizer.alignCrop(frame, face)
            emb = recognizer.feature(aligned)

            best_similarity = -1.0
            best_person_id = -1

            # Match against database embeddings using Cosine Similarity
            for person_id, stored_embs in database_embeddings.items():
                for stored_emb in stored_embs:
                    similarity = recognizer.match(emb, stored_emb, cv2.FaceRecognizerSF_FR_COSINE)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_person_id = person_id

            # Dynamic cache update if ID from DB is not loaded
            if best_person_id != -1 and best_person_id not in people:
                cursor.execute("SELECT name, person_type, department FROM staff WHERE id=%s", (best_person_id,))
                row = cursor.fetchone()
                if row:
                    people[best_person_id] = (row[0], row[1], row[2])

            if DEBUG_PRINT:
                known_name = people[best_person_id][0] if best_person_id in people else "not-in-db"
                print(f"predicted_id={best_person_id} ({known_name})  similarity={best_similarity:.3f}  threshold={CONFIDENCE_THRESHOLD}")

            # Match criteria: similarity must be >= threshold
            raw_result = best_person_id if (best_similarity >= CONFIDENCE_THRESHOLD and best_person_id in people) else "unknown"
            
            bucket = get_position_bucket(x, y)
            active_buckets_this_frame.add(bucket)

            # --- Liveness Detection Gate ---
            # Check if this face box overlaps with any detected screen box by > 35%
            face_overlaps_screen = False
            for s_box in detected_screens:
                overlap = calculate_overlap_ratio((x, y, w, h), s_box[:4])
                if overlap > 0.35:
                    face_overlaps_screen = True
                    break

            if face_overlaps_screen:
                liveness_status = "spoof_phone"
            elif raw_result == "unknown":
                # Bypass liveness check for unknown visitors to make logs instant
                liveness_status = "unknown_visitor"
            else:
                landmarks = face_utils.get_face_mesh_landmarks(face_crop)
                liveness_status = "verifying"
                if landmarks is not None:
                    liveness_status = check_liveness_state(bucket, landmarks, face_crop, raw_result)
                else:
                    if bucket in liveness_history:
                        liveness_history[bucket]["consecutive_frames"] += 1
                        if liveness_history[bucket]["consecutive_frames"] > 80:
                            liveness_status = "spoof"

            if liveness_status == "passed":
                liveness_result = raw_result
                # Clear 'verifying' tokens from queue so recognition triggers instantly
                if bucket in prediction_history and "verifying" in prediction_history[bucket]:
                    prediction_history[bucket].clear()
            elif liveness_status == "spoof":
                liveness_result = "spoof"
            elif liveness_status == "spoof_phone":
                liveness_result = "spoof_phone"
                if bucket in liveness_history:
                    if not liveness_history[bucket].get("spoof_alert_logged", False) and raw_result != "unknown":
                        log_spoof(face_crop)
                        liveness_history[bucket]["spoof_alert_logged"] = True
            elif liveness_status == "unknown_visitor":
                liveness_result = "unknown"
            elif liveness_status == "look_at_camera":
                liveness_result = "look_at_camera"
            else:
                liveness_result = "verifying"

            stable_result = get_stable_prediction(bucket, liveness_result)

            now_ts = datetime.datetime.now().timestamp()

            if stable_result == "verifying":
                label = "Verifying liveness..."
                color = (0, 200, 255)

            elif stable_result == "look_at_camera":
                label = "Please look at the camera!"
                color = (0, 165, 255)

            elif stable_result == "spoof":
                label = "Spoof suspected"
                color = (0, 0, 255)

            elif stable_result == "spoof_phone":
                label = "Spoof: Phone/Proximity"
                color = (0, 0, 255)

            elif stable_result != "unknown":
                name, person_type, department = people[stable_result]
                label = "Present"
                color = (0, 255, 0)

                if now_ts - last_marked.get(stable_result, 0) > MARK_COOLDOWN:
                    mark_attendance(stable_result)
                    last_marked[stable_result] = now_ts
                    print(f"Attendance marked: {name} ({person_type}, {department})")

            else:
                label = "Unknown"
                color = (0, 0, 255)

                if now_ts - last_marked.get("unknown", 0) > MARK_COOLDOWN:
                    log_unknown(face_crop)
                    last_marked["unknown"] = now_ts

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if not face_processed:
        msg = "Face-ah correct-ah kaatunga! / Please show face!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255) # White
        thickness = 2
        text_size = cv2.getTextSize(msg, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = 45
        
        # Semi-transparent dark background header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, msg, (text_x, text_y), font, font_scale, color, thickness)

    # Clean up stale liveness trackers
    stale_buckets = []
    for b in list(liveness_history.keys()):
        if b not in active_buckets_this_frame:
            if "missing_frames" not in liveness_history[b]:
                liveness_history[b]["missing_frames"] = 0
            liveness_history[b]["missing_frames"] += 1
            if liveness_history[b]["missing_frames"] > 30:
                stale_buckets.append(b)
        else:
            liveness_history[b]["missing_frames"] = 0
            
    for b in stale_buckets:
        liveness_history.pop(b, None)
        prediction_history.pop(b, None)
        last_stable_result.pop(b, None)

    cv2.imshow('Live Attendance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

export_attendance_to_excel()
conn.close()