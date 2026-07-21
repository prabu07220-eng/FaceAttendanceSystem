from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import mysql.connector
import datetime
import openpyxl
import io
import os
try:
    import cv2
except Exception as _cv_err:
    print(f"Notice: Running in headless web mode without OpenCV GUI: {_cv_err}")
    cv2 = None

import hashlib
import secrets
import subprocess

try:
    import face_utils
except Exception as _fu_err:
    print(f"Notice: face_utils import fallback: {_fu_err}")
    face_utils = None

import numpy as np
import config

import jinja2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)
app.jinja_loader = jinja2.ChoiceLoader([
    jinja2.FileSystemLoader(os.path.join(BASE_DIR, 'templates')),
    jinja2.FileSystemLoader(BASE_DIR)
])

@app.route('/style.css')
def serve_style_css():
    from flask import send_from_directory
    for folder in [os.path.join(BASE_DIR, 'static'), BASE_DIR]:
        if os.path.exists(os.path.join(folder, 'style.css')):
            return send_from_directory(folder, 'style.css', mimetype='text/css')
    return "", 404

@app.route('/static/<path:filename>', endpoint='static')
def custom_static(filename):
    from flask import send_from_directory
    for folder in [os.path.join(BASE_DIR, 'static'), BASE_DIR]:
        if os.path.exists(os.path.join(folder, filename)):
            return send_from_directory(folder, filename)
    return "", 404

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    print("SERIOUS SERVER ERROR:\n", tb)
    return f"<h3>Server Debug Error</h3><pre>{tb}</pre>", 500

# Secure persistent secret key generation
SECRET_KEY_PATH = os.path.join(BASE_DIR, ".secret_key")
if os.path.exists(SECRET_KEY_PATH):
    try:
        with open(SECRET_KEY_PATH, "r") as f:
            app.secret_key = f.read().strip()
    except Exception:
        app.secret_key = secrets.token_hex(32)
else:
    try:
        key = secrets.token_hex(32)
        with open(SECRET_KEY_PATH, "w") as f:
            f.write(key)
        app.secret_key = key
    except Exception:
        app.secret_key = secrets.token_hex(32)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_DIR = os.path.join(BASE_DIR, "trainer")
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")
FRONTAL_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
PROFILE_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_profileface.xml')

# Tracks the running recognize_attendance.py process (None = not running)
recognition_process = None

# Hashed admin credentials (SHA-256 hashes of 'admin123' and 'register123')
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"
REGISTER_PASSWORD_HASH = "ea8a78bdd077d5303572bf868580be3668cba7cdd0c151ccd309afed777f37e0"

TARGET_IMAGES = 100
CAPTURE_DELAY = 0.15


# Database helper function
def get_db():
    return config.get_db_connection()


@app.context_processor
def inject_settings():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT `key`, `value` FROM settings")
        settings_dict = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        chat_enabled = settings_dict.get("ai_chat_enabled", "true") == "true"
    except Exception:
        chat_enabled = True
    return dict(ai_chat_enabled=chat_enabled)


def login_required(func):
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    if request.method == "POST":
        entered_password = request.form.get("register_password", "")
        entered_hash = hashlib.sha256(entered_password.encode()).hexdigest()
        if entered_hash != REGISTER_PASSWORD_HASH:
            flash("Incorrect registration password. Person was not added.", "error")
            return redirect(url_for("register"))

        name = request.form.get("name", "").strip()
        person_type = request.form.get("person_type", "Student")
        id_number = request.form.get("id_number", "").strip()
        department = request.form.get("department", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()

        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO staff (name, person_type, id_number, department, email, phone) VALUES (%s, %s, %s, %s, %s, %s)",
                (name, person_type, id_number, department, email, phone)
            )
            conn.commit()
        except mysql.connector.IntegrityError:
            conn.close()
            flash(f"{person_type} ID '{id_number}' is already registered. Please use a different ID.", "error")
            return redirect(url_for("register"))

        new_id = cursor.lastrowid
        conn.close()

        # Launch the webcam to capture face images for this person
        captured = capture_face_images(new_id, name)

        if captured > 0:
            flash(f"{person_type} '{name}' added successfully with {captured} face images captured.", "success")
        else:
            flash(f"{person_type} '{name}' was added to the database, but no face images were captured "
                  f"(camera issue?). You can register their face later using capture_faces.py.", "error")
        return redirect(url_for("people"))

    return render_template("register.html")


def capture_face_images(person_id, name):
    """Opens the webcam and captures face images for a newly registered person."""
    if cv2 is None or face_utils is None:
        return 0
    face_dir = os.path.join(DATASET_DIR, f"{person_id}_{name}")
    os.makedirs(face_dir, exist_ok=True)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, test_frame = cam.read()
    if not ret:
        cam.release()
        return 0
    
    height, width = test_frame.shape[:2]
    detector = face_utils.get_yunet_detector(width, height)
    recognizer = face_utils.get_sface_recognizer()

    count = 0
    last_save_time = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        detector.setInputSize((frame.shape[1], frame.shape[0]))
        retval, faces = detector.detect(frame)

        now = datetime.datetime.now().timestamp()
        if faces is not None and len(faces) > 0:
            # Find and capture the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            
            # Draw green rectangle for the target face
            x, y, w, h = map(int, largest_face[0:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw red rectangle for any other faces in the background
            for face in faces:
                if not np.array_equal(face, largest_face):
                    ox, oy, ow, oh = map(int, face[0:4])
                    cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 1)

            if now - last_save_time >= CAPTURE_DELAY:
                count += 1
                aligned_face = recognizer.alignCrop(frame, largest_face)
                cv2.imwrite(os.path.join(face_dir, f"{count}.jpg"), aligned_face)
                last_save_time = now

        cv2.putText(frame, f"Captured: {count}/{TARGET_IMAGES}  (press q to stop)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Registering Face - Press q to stop', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= TARGET_IMAGES:
            break

    cam.release()
    cv2.destroyAllWindows()
    return count


@app.route("/retrain")
@login_required
def retrain():
    import pickle
    
    # Get active IDs from the database to avoid training on orphaned folders
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM staff")
    valid_ids = {row[0] for row in cursor.fetchall()}
    conn.close()

    recognizer = face_utils.get_sface_recognizer()
    
    # We will save the embeddings as a dictionary: {person_id: [embedding1, embedding2, ...]}
    database_embeddings = {}
    total_images = 0

    # Ensure dataset directory exists
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)

    for folder_name in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        try:
            person_id = int(folder_name.split("_")[0])
            if person_id not in valid_ids:
                continue
        except ValueError:
            continue

        person_embeddings = []
        for image_name in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, image_name))
            if img is not None:
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

    if total_images == 0:
        flash("No valid face images found to train on.", "error")
    else:
        os.makedirs(TRAINER_DIR, exist_ok=True)
        pickle_path = os.path.join(TRAINER_DIR, "embeddings.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(database_embeddings, f)
        flash(f"Model retrained successfully on {total_images} images.", "success")

    return redirect(url_for("people"))


@app.route("/people/delete/<int:person_id>", methods=["POST"])
@login_required
def delete_person(person_id):
    import shutil

    conn = get_db()
    cursor = conn.cursor()

    # Get name so we can find their dataset folder
    cursor.execute("SELECT name FROM staff WHERE id=%s", (person_id,))
    row = cursor.fetchone()

    if row is None:
        conn.close()
        flash("Person not found.", "error")
        return redirect(url_for("people"))

    name = row[0]

    # Delete dependent attendance records first (foreign key)
    cursor.execute("DELETE FROM attendance WHERE staff_id=%s", (person_id,))
    # Delete the staff/student record itself
    cursor.execute("DELETE FROM staff WHERE id=%s", (person_id,))
    conn.commit()
    conn.close()

    # Delete their captured face images from disk
    face_dir = os.path.join(DATASET_DIR, f"{person_id}_{name}")
    if os.path.isdir(face_dir):
        shutil.rmtree(face_dir)

    flash(f"'{name}' and all related records (attendance, face photos) were deleted. "
          f"Click 'Retrain Model' so the recognizer forgets their face.", "success")
    return redirect(url_for("people"))


@app.route("/alerts/delete/<int:alert_id>", methods=["POST"])
@login_required
def delete_alert(alert_id):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT image_path FROM unknown_alerts WHERE id=%s", (alert_id,))
    row = cursor.fetchone()

    if row is not None:
        image_path = row[0]
        cursor.execute("DELETE FROM unknown_alerts WHERE id=%s", (alert_id,))
        conn.commit()
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        flash("Alert deleted.", "success")
    else:
        flash("Alert not found.", "error")

    conn.close()
    return redirect(url_for("alerts"))


@app.route("/alerts/delete_all", methods=["POST"])
@login_required
def delete_all_alerts():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM unknown_alerts")
    paths = [row[0] for row in cursor.fetchall()]

    cursor.execute("DELETE FROM unknown_alerts")
    conn.commit()
    conn.close()

    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)

    flash("All unknown alerts deleted.", "success")
    return redirect(url_for("alerts"))


@app.route("/alerts/delete_spoof/<int:alert_id>", methods=["POST"])
@login_required
def delete_spoof(alert_id):
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT image_path FROM spoof_alerts WHERE id=%s", (alert_id,))
    row = cursor.fetchone()

    if row is not None:
        image_path = row[0]
        cursor.execute("DELETE FROM spoof_alerts WHERE id=%s", (alert_id,))
        conn.commit()
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        flash("Spoof alert deleted.", "success")
    else:
        flash("Spoof alert not found.", "error")

    conn.close()
    return redirect(url_for("alerts"))


@app.route("/alerts/delete_all_spoofs", methods=["POST"])
@login_required
def delete_all_spoofs():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM spoof_alerts")
    paths = [row[0] for row in cursor.fetchall()]

    cursor.execute("DELETE FROM spoof_alerts")
    conn.commit()
    conn.close()

    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)

    flash("All spoof alerts cleared.", "success")
    return redirect(url_for("alerts"))


@app.route("/toggle_recognition", methods=["POST"])
@login_required
def toggle_recognition():
    global recognition_process

    if recognition_process is not None and recognition_process.poll() is None:
        # Currently running -> stop it
        recognition_process.terminate()
        recognition_process = None
        flash("Live recognition stopped.", "success")
    else:
        # Not running -> start it as a separate process (opens its own camera window)
        script_path = os.path.join(BASE_DIR, "recognize_attendance.py")
        recognition_process = subprocess.Popen(["python", script_path], cwd=BASE_DIR)
        flash("Live recognition started. A camera window should open shortly.", "success")

    return redirect(url_for("dashboard"))


@app.route("/recognition_status")
@login_required
def recognition_status():
    running = recognition_process is not None and recognition_process.poll() is None
    return {"running": running}


@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        entered_hash = hashlib.sha256(password.encode()).hexdigest()
        if username.lower() == ADMIN_USERNAME.lower() and entered_hash == ADMIN_PASSWORD_HASH:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Incorrect username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    student_count, staff_count, present_today, alerts_today = 0, 0, 0, 0
    recent = []
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM staff WHERE person_type='Student'")
        student_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM staff WHERE person_type='Staff'")
        staff_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT staff_id) FROM attendance WHERE DATE(entry_time)=%s", (datetime.date.today(),))
        present_today = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM unknown_alerts WHERE DATE(detected_time)=%s", (datetime.date.today(),))
        alerts_today = cursor.fetchone()[0]

        cursor.execute("""
            SELECT s.name, s.person_type, s.department, a.entry_time
            FROM attendance a JOIN staff s ON a.staff_id = s.id
            ORDER BY a.entry_time DESC LIMIT 8
        """)
        recent = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Notice: dashboard DB fallback: {e}")

    return render_template(
        "dashboard.html",
        student_count=student_count,
        staff_count=staff_count,
        present_today=present_today,
        alerts_today=alerts_today,
        recent=recent
    )


@app.route("/people")
@login_required
def people():
    rows, departments = [], []
    dept_filter = request.args.get("department", "")
    type_filter = request.args.get("type", "")

    try:
        conn = get_db()
        cursor = conn.cursor()

        query = "SELECT id, name, person_type, id_number, department, email, phone FROM staff WHERE 1=1"
        params = []
        if dept_filter:
            query += " AND department = %s"
            params.append(dept_filter)
        if type_filter:
            query += " AND person_type = %s"
            params.append(type_filter)
        query += " ORDER BY name"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        cursor.execute("SELECT DISTINCT department FROM staff WHERE department IS NOT NULL AND department <> ''")
        departments = [r[0] for r in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"Notice: people DB fallback: {e}")

    return render_template("people.html", rows=rows, departments=departments,
                            dept_filter=dept_filter, type_filter=type_filter)


@app.route("/attendance")
@login_required
def attendance():
    rows = []
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id_number, s.name, s.person_type, s.department, a.entry_time, a.exit_time
            FROM attendance a JOIN staff s ON a.staff_id = s.id
            ORDER BY a.entry_time DESC
        """)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Notice: attendance DB fallback: {e}")

    return render_template("attendance.html", rows=rows)


@app.route("/attendance/download")
@login_required
def download_attendance():
    rows = []
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id_number, s.name, s.person_type, s.department, a.entry_time, a.exit_time
            FROM attendance a JOIN staff s ON a.staff_id = s.id
            ORDER BY a.entry_time DESC
        """)
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Notice: download DB fallback: {e}")

    wb = openpyxl.Workbook()
    headers = ["ID Number", "Name", "Type", "Department", "Status", "Entry Time", "Exit Time"]

    ws_students = wb.active
    ws_students.title = "Students"
    ws_students.append(headers)

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

    for col_cells in ws_students.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col_cells)
        ws_students.column_dimensions[col_cells[0].column_letter].width = max_len + 3

    for col_cells in ws_staff.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col_cells)
        ws_staff.column_dimensions[col_cells[0].column_letter].width = max_len + 3

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="attendance_report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/alerts")
@login_required
def alerts():
    unknown_rows, spoof_rows = [], []
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, detected_time, image_path FROM unknown_alerts ORDER BY detected_time DESC")
        unknown_rows = cursor.fetchall()
        cursor.execute("SELECT id, detected_time, image_path FROM spoof_alerts ORDER BY detected_time DESC")
        spoof_rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Notice: alerts DB fallback: {e}")

    return render_template("alerts.html", active="alerts", unknown_rows=unknown_rows, spoof_rows=spoof_rows)


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    llm_provider = "gemini"
    gemini_api_key = ""
    ollama_model = "qwen2.5-coder:1.5b"
    ai_chat_enabled = "true"

    try:
        conn = get_db()
        cursor = conn.cursor()
        
        if request.method == "POST":
            provider = request.form.get("llm_provider", "gemini").strip()
            api_key = request.form.get("gemini_api_key", "").strip()
            model_name = request.form.get("ollama_model", "qwen2.5-coder:1.5b").strip()
            chat_enabled = request.form.get("ai_chat_enabled", "true").strip()
            
            cursor.execute("INSERT INTO settings (`key`, `value`) VALUES ('llm_provider', %s) ON DUPLICATE KEY UPDATE `value` = %s", (provider, provider))
            cursor.execute("INSERT INTO settings (`key`, `value`) VALUES ('gemini_api_key', %s) ON DUPLICATE KEY UPDATE `value` = %s", (api_key, api_key))
            cursor.execute("INSERT INTO settings (`key`, `value`) VALUES ('ollama_model', %s) ON DUPLICATE KEY UPDATE `value` = %s", (model_name, model_name))
            cursor.execute("INSERT INTO settings (`key`, `value`) VALUES ('ai_chat_enabled', %s) ON DUPLICATE KEY UPDATE `value` = %s", (chat_enabled, chat_enabled))
            conn.commit()
            flash("Settings saved successfully.", "success")
            
        cursor.execute("SELECT `key`, `value` FROM settings")
        settings_dict = {row[0]: row[1] for row in cursor.fetchall()}
        
        llm_provider = settings_dict.get("llm_provider", "gemini")
        gemini_api_key = settings_dict.get("gemini_api_key", "")
        ollama_model = settings_dict.get("ollama_model", "qwen2.5-coder:1.5b")
        ai_chat_enabled = settings_dict.get("ai_chat_enabled", "true")
        conn.close()
    except Exception as e:
        print(f"Notice: settings DB fallback: {e}")

    return render_template(
        "settings.html",
        active="settings",
        llm_provider=llm_provider,
        gemini_api_key=gemini_api_key,
        ollama_model=ollama_model,
        ai_chat_enabled=ai_chat_enabled
    )


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    data = request.get_json() or {}
    user_question = data.get("message", "").strip()
    if not user_question:
        return {"response": "Please enter a question."}

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT `key`, `value` FROM settings")
    settings_dict = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    llm_provider = settings_dict.get("llm_provider", "gemini")
    gemini_api_key = settings_dict.get("gemini_api_key", "")
    ollama_model = settings_dict.get("ollama_model", "qwen2.5-coder:1.5b")

    # The system instruction prompt for SQL generation
    prompt_sql = f"""
    You are an assistant for a school/office attendance database. 
    Write a single read-only MySQL SELECT query that answers this question: "{user_question}".
    
    IMPORTANT:
    1. If the user's message is a greeting (like hello, hi, hey, greetings, good morning), thank you, or general chit-chat that does not require database data, return exactly "NO_QUERY" (no quotes, no other text).
    2. Return ONLY the raw SQL query, no markdown block (do NOT wrap it in ```sql), no quotes, no explanation.
    3. Ensure the query is read-only (SELECT queries only).
    4. The time threshold for latecomers is 9:30 AM (i.e. TIME(entry_time) > '09:30:00').
    
    Database Schema:
    - staff(id, name, person_type, id_number, department, email, phone, reg_date)
      Note: person_type is ENUM('Student', 'Staff')
    - attendance(id, staff_id, entry_time, exit_time)
    - unknown_alerts(id, detected_time, image_path)
    """

    sql_query = ""
    gemini_model_instance = None

    if llm_provider == "gemini":
        if not gemini_api_key:
            return {"response": "Please configure your Gemini API Key in the Settings tab to activate the AI Admin Assistant."}
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            gemini_model_instance = genai.GenerativeModel('gemini-1.5-flash')
            response_sql = gemini_model_instance.generate_content(prompt_sql)
            sql_query = response_sql.text.strip()
        except Exception as e:
            return {"response": f"Gemini API Error: {e}"}
    else:
        # Local Ollama
        import requests
        try:
            payload = {
                "model": ollama_model,
                "messages": [{"role": "user", "content": prompt_sql}],
                "stream": False
            }
            res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=20)
            if res.status_code == 200:
                sql_query = res.json()["message"]["content"].strip()
            else:
                raise Exception(f"Ollama HTTP {res.status_code}")
        except Exception as e:
            return {"response": "Ollama is not running locally. Please start the Ollama application on your computer and run 'ollama run qwen2.5-coder:1.5b' in your terminal."}

    # Defensive clean of markdown wrappers
    if sql_query.startswith("```"):
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    is_no_query = sql_query.strip().upper() == "NO_QUERY"
    
    if is_no_query:
        query_success = True
        sql_query = "" # Clear it so no SQL card displays in UI
        results_str = "None"
        prompt_summary = f"""
        The user said: "{user_question}"
        Please respond to them in a friendly, conversational way. You do not need to query the database or return any rows.
        """
    else:
        # 2. Execute the query
        conn = get_db()
        cursor = conn.cursor()
        try:
            if ";" in sql_query:
                raise ValueError("Multiple SQL statements (semicolon) are not allowed.")

            if not sql_query.upper().strip().startswith("SELECT"):
                raise ValueError("Only read-only SELECT queries are allowed.")
                
            cursor.execute(sql_query)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            
            formatted_rows = []
            for r in rows:
                formatted_row = {}
                for idx, col_name in enumerate(columns):
                    val = r[idx]
                    if isinstance(val, (datetime.datetime, datetime.date)):
                        val = val.strftime("%Y-%m-%d %H:%M:%S")
                    formatted_row[col_name] = val
                formatted_rows.append(formatted_row)
                
            query_success = True
            results_str = str(formatted_rows)
        except Exception as sql_err:
            query_success = False
            results_str = f"Error executing query: {sql_err}"
        finally:
            conn.close()

        # 3. Format conversational response
        if query_success:
            prompt_summary = f"""
            The user asked: "{user_question}"
            The database returned these rows: {results_str}
            The SQL query run was: {sql_query}
            
            Please format a natural, friendly, and concise response to the user.
            If the database returned no rows, explain that no matching records were found.
            """
        else:
            prompt_summary = f"""
            The user asked: "{user_question}"
            The system tried to execute this query: {sql_query}
            But it failed with this error: {results_str}
            
            Explain the failure to the user in a friendly way and suggest how they might rephrase it.
            """

    reply = ""
    if llm_provider == "gemini":
        try:
            response_summary = gemini_model_instance.generate_content(prompt_summary)
            reply = response_summary.text.strip()
        except Exception as e:
            reply = f"Failed to format response with Gemini: {e}. Raw results: {results_str}"
    else:
        # Local Ollama
        import requests
        try:
            payload = {
                "model": ollama_model,
                "messages": [{"role": "user", "content": prompt_summary}],
                "stream": False
            }
            res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=25)
            if res.status_code == 200:
                reply = res.json()["message"]["content"].strip()
            else:
                raise Exception(f"Ollama HTTP {res.status_code}")
        except Exception as e:
            reply = f"Query executed successfully, but Ollama failed to summarize: {e}. Raw results: {results_str}"

    return {"response": reply, "sql": sql_query}


@app.route("/analytics")
@login_required
def analytics():
    hourly_labels, hourly_data = [], []
    late_labels, late_data = [], []
    trend_labels, trend_rates = [], []
    predicted_rate = 0.0
    forecast_status = "No attendance logs found yet. Register people and log entry to calculate."

    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # 1. Total Registered
        cursor.execute("SELECT COUNT(*) FROM staff")
        total_registered = cursor.fetchone()[0] or 1 # Avoid division by zero
        
        # 2. Busiest Hours (Hourly Arrivals)
        cursor.execute("""
            SELECT HOUR(entry_time) as hr, COUNT(*) as cnt 
            FROM attendance 
            GROUP BY HOUR(entry_time) 
            ORDER BY hr
        """)
        hourly_rows = cursor.fetchall()
        for hr, cnt in hourly_rows:
            ampm = "AM" if hr < 12 else "PM"
            display_hr = hr if hr <= 12 else hr - 12
            if display_hr == 0:
                display_hr = 12
            hourly_labels.append(f"{display_hr:02d}:00 {ampm}")
            hourly_data.append(cnt)
            
        # 3. Late Arrivals after 9:30 AM by Department
        cursor.execute("""
            SELECT COALESCE(s.department, 'No Dept') as dept, COUNT(*) as cnt 
            FROM attendance a 
            JOIN staff s ON a.staff_id = s.id 
            WHERE TIME(a.entry_time) > '09:30:00' 
            GROUP BY s.department
        """)
        late_rows = cursor.fetchall()
        late_labels = [row[0] for row in late_rows]
        late_data = [row[1] for row in late_rows]
        
        # 4. Attendance Trends (Past 30 Days)
        cursor.execute("""
            SELECT DATE(entry_time) as dt, COUNT(DISTINCT staff_id) as cnt 
            FROM attendance 
            WHERE entry_time >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) 
            GROUP BY DATE(entry_time) 
            ORDER BY dt
        """)
        trend_rows = cursor.fetchall()
        for dt, cnt in trend_rows:
            date_str = dt.strftime("%b %d")
            rate = round((cnt / total_registered) * 100, 1)
            trend_labels.append(date_str)
            trend_rates.append(rate)
            
        # 5. AI Attendance Predictor (Weighted Moving Average)
        cursor.execute("""
            SELECT DATE(entry_time) as dt, COUNT(DISTINCT staff_id) as cnt 
            FROM attendance 
            GROUP BY DATE(entry_time) 
            ORDER BY dt DESC 
            LIMIT 14
        """)
        predictor_rows = cursor.fetchall()
        predictor_rows.reverse()
        
        if len(predictor_rows) >= 3:
            rates = [round((cnt / total_registered) * 100, 1) for dt, cnt in predictor_rows]
            weighted_sum = 0.0
            weight_total = 0.0
            for idx, rate in enumerate(rates):
                weight = idx + 1
                weighted_sum += rate * weight
                weight_total += weight
            predicted_rate = round(weighted_sum / weight_total, 1)
            forecast_status = f"Based on last {len(rates)} days of attendance logs."
        elif len(predictor_rows) > 0:
            rates = [round((cnt / total_registered) * 100, 1) for dt, cnt in predictor_rows]
            predicted_rate = round(sum(rates) / len(rates), 1)
            forecast_status = "Based on limited attendance history."
        conn.close()
    except Exception as e:
        print(f"Notice: analytics DB fallback: {e}")
        
    return render_template(
        "analytics.html",
        active="analytics",
        hourly_labels=hourly_labels,
        hourly_data=hourly_data,
        late_labels=late_labels,
        late_data=late_data,
        trend_labels=trend_labels,
        trend_rates=trend_rates,
        predicted_rate=predicted_rate,
        forecast_status=forecast_status
    )


if __name__ == "__main__":
    try:
        from waitress import serve
        print("Starting Waitress WSGI production server on http://127.0.0.1:5000...")
        serve(app, host="127.0.0.1", port=5000, threads=8)
    except ImportError:
        app.run(debug=True, load_dotenv=False)
