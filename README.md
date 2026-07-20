# Face Attendance System (Prototype)

Based on: Person Re-Identification and Time Tracking Using Face Features

## Setup Steps

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Start XAMPP -> turn ON Apache + MySQL.

3. Open http://localhost/phpmyadmin and import `schema.sql`
   (or paste its contents into the SQL tab and run it).

4. Register a person (Student or Staff) and capture their face:
   ```
   python capture_faces.py
   ```
   - Choose 1 (Student) or 2 (Staff) when asked.
   - Enter name, ID number (roll no / staff ID), email, phone.
   - Webcam will open — show your face at different angles.
   - It auto-stops after 50 images (or press 'q' to stop early).
   - Repeat this for every student/staff member you want to register.

## Folder structure

```
FaceAttendanceSystem/
 ├── dataset/       -> captured face images (auto-created per staff)
 ├── trainer/       -> trained recognition model will be saved here (next step)
 ├── static/        -> Flask static files (next step)
 ├── templates/     -> Flask HTML pages (next step)
 ├── capture_faces.py
 ├── schema.sql
 ├── requirements.txt
 └── README.md
```

## AI Face Liveness & Anti-Spoofing Layer

We have implemented a real-time, CPU-optimized **Face Liveness & Anti-Spoofing** layer using **MediaPipe Face Mesh**. It runs directly on the cropped face ROI (Region of Interest) to minimize CPU overhead and keep frame rates high.

### How it Works:
1. **Blink Verification**: Calculates the Eye Aspect Ratio (EAR). The system requires at least one natural blink (EAR dropping below `0.18` and returning above `0.24`) to confirm a live subject.
2. **Micro-movement / 3D Coordinate Check**: Tracks the variance of normalized landmarks. Flat screens or printed photos show rigid planar landmarks and will fail this non-rigid coordinate variance check.
3. **UX Fallback**: If no head turn is detected, a blink-only fallback triggers after 60 frames to keep check-ins seamless.
4. **Spoof Alerts**: If verification fails for 80 frames (approx. 3-4 seconds), a red box says "Spoof suspected" and log snapshots are sent to `static/spoof_attempts/` and saved to the `spoof_alerts` database table for admin review on the dashboard.

### How to Tune Thresholds:
You can tune the liveness variables inside `recognize_attendance.py`:
- `ear < 0.18` (Closed eye EAR threshold) and `ear > 0.24` (Open eye EAR threshold).
- `mean_std > 0.0003` (Micro-movement standard deviation threshold). Increase this value to require more head movement, or lower it if users complain about too strict checks.
- `consecutive_frames > 80` (Number of frames before spoof is flagged). Increase to give users more time to blink/verify.

## Flask Web Dashboard

```
python app.py
```
Open http://localhost:5000 in your browser.

Default login: **admin / admin123**

Pages:
- **Overview** — quick stats and recent activity
- **Students & Staff** — directory with type/department filters, "+ Add Person" button, "Retrain Model" button
- **Add Person** — register a new student/staff from the browser (requires the registration password below), opens the webcam automatically to capture their face
- **Attendance Log** — full log with an Excel download button
- **Unknown Alerts** — gallery of unrecognized face detections

Registration password (needed on the Add Person page): **register123**

After adding people through the dashboard, click **Retrain Model** on the Students & Staff page so the recognizer learns their face.
