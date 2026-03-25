# AFAS – Automatic Face Attendance System

AFAS (Automatic Face Attendance System) is a full-stack attendance management application that uses face recognition to automate attendance marking. The system is designed to reduce manual attendance errors, improve speed, and provide a more modern, contactless attendance process for academic environments.

This project combines a desktop-based user interface for registration and attendance scanning with a web-based backend and admin portal for managing users, logs, and attendance records.

## Features

- Face-based attendance marking
- User registration with facial sample enrollment
- Student and teacher role support
- Automatic attendance logging
- Duplicate face detection during enrollment
- Teacher/admin portal for system monitoring
- Attendance history view
- CSV export for attendance records
- Liveness / anti-spoofing related improvements in progress
- Dashboard summaries for users and attendance data

## Project Structure

```bash
AFAS/
│
├── desktop/
│   ├── main.py
│   ├── home_screen.py
│   ├── register_screen.py
│   ├── attendance_screen.py
│   ├── api_client.py
│   └── camera_utils.py
│
├── server/
│   ├── app.py
│   ├── models.py
│   ├── auth.py
│   ├── routes/
│   ├── templates/
│   ├── static/
│   └── instance/
│
├── .gitignore
└── README.md
