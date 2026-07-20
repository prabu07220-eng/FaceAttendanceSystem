CREATE DATABASE IF NOT EXISTS face_attendance_db2;
USE face_attendance_db2;

CREATE TABLE staff (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    person_type ENUM('Student', 'Staff') NOT NULL,
    id_number VARCHAR(50) UNIQUE,
    department VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(15),
    reg_date DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    staff_id INT,
    entry_time DATETIME,
    exit_time DATETIME,
    FOREIGN KEY (staff_id) REFERENCES staff(id)
);

CREATE TABLE unknown_alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detected_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS settings (
    `key` VARCHAR(255) PRIMARY KEY,
    `value` TEXT
);

CREATE TABLE IF NOT EXISTS spoof_alerts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    detected_time DATETIME NOT NULL,
    image_path VARCHAR(255) NOT NULL
);

