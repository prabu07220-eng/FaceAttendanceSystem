import cv2
import os
import time
import mysql.connector
import face_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# MySQL connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="face_attendance_db2"
)
cursor = conn.cursor()

# TARGET_IMAGES to capture
TARGET_IMAGES = 100
CAPTURE_DELAY = 0.15

TARGET_IMAGES = 100      # total images to capture per person (more angles = better recognition)
CAPTURE_DELAY = 0.15     # seconds between saved frames, gives time to turn your head

while True:
    # Person details input
    print("\nRegister as:")
    print("1. Student")
    print("2. Staff")
    choice = input("Enter 1 or 2: ").strip()
    person_type = "Student" if choice == "1" else "Staff"

    name = input("Enter name: ")

    # Keep asking for ID number until a unique one is given
    while True:
        id_number = input(f"Enter {person_type} ID: ")
        department = input("Enter Department: ")
        email = input("Enter email: ")
        phone = input("Enter phone: ")

        try:
            cursor.execute(
                "INSERT INTO staff (name, person_type, id_number, department, email, phone) VALUES (%s, %s, %s, %s, %s, %s)",
                (name, person_type, id_number, department, email, phone)
            )
            conn.commit()
            break  # insert successful, move on
        except mysql.connector.IntegrityError:
            conn.rollback()
            print(f"\n{person_type} ID '{id_number}' is already registered. Please enter a different ID.\n")

    staff_id = cursor.lastrowid
    print(f"{person_type} registered with ID: {staff_id}")

    # Create folder for this person's face images
    face_dir = os.path.join(DATASET_DIR, f"{staff_id}_{name}")
    os.makedirs(face_dir, exist_ok=True)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, test_frame = cam.read()
    if not ret:
        print("Could not open the camera. Please check that your webcam is connected.")
        cam.release()
        continue
    
    height, width = test_frame.shape[:2]
    detector = face_utils.get_yunet_detector(width, height)
    recognizer = face_utils.get_sface_recognizer()

    count = 0
    last_save_time = 0

    print("Look at the camera. Slowly turn your head left, then right, then straight.")
    print(f"Capturing {TARGET_IMAGES} images. Press 'q' to stop early.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Set detector input size in case of resolution changes
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        retval, faces = detector.detect(frame)

        now = time.time()
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

        cv2.putText(frame, f"Captured: {count}/{TARGET_IMAGES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Capturing Faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count >= TARGET_IMAGES:
            break

    print(f"{count} images captured for {name}")
    cam.release()
    cv2.destroyAllWindows()

    again = input("\nAdd another person? (y/n): ").strip().lower()
    if again != "y":
        break

conn.close()
print("Registration process completed. Thank you!")
