# AFAS - Automatic Face Attendance System

AFAS is a full-stack face-recognition attendance project built for registering users and marking attendance automatically. The system is split into two connected applications:

- `server/` contains the Flask backend, database models, face-matching logic, and web portal.
- `desktop/` contains the Tkinter desktop client used for user registration and attendance scanning.

The project is designed so the desktop app captures faces and sends them to the backend, while the server stores users, compares face embeddings, records attendance, and provides an admin/teacher management portal.

## Features

- Face-based attendance marking
- User registration with face enrollment
- Student and teacher support
- Duplicate face detection during enrollment
- Attendance history and dashboard reporting
- CSV export for attendance records
- Admin and teacher web portal
- Password reset with OTP email flow
- Basic liveness-oriented checks during registration

## Project Structure

```text
AFAS/
|-- desktop/
|   |-- main.py
|   |-- home_screen.py
|   |-- register_screen.py
|   |-- attendance_screen.py
|   |-- api_client.py
|   |-- camera_utils.py
|   `-- ui_theme.py
|
|-- server/
|   |-- app.py
|   |-- models.py
|   |-- auth.py
|   |-- templates/
|   |-- static/
|   `-- instance/
|
`-- README.md
```

## Tech Stack

### Server

- Flask
- Flask-SQLAlchemy
- SQLite
- OpenCV
- DeepFace
- NumPy
- Waitress

### Desktop

- Tkinter
- OpenCV
- MediaPipe
- Pillow
- Requests

## How It Works

1. A user is created from the desktop registration screen.
2. The desktop app captures a face sample and sends it to the server.
3. The server stores the user, face sample, and embedding.
4. During attendance, the desktop scanner sends live camera frames to the backend.
5. The backend compares the face against enrolled users and marks attendance if a valid match is found.
6. Admins and teachers can view attendance, manage users, and export records from the browser portal.

## Running the Project

This project should be run using two separate folders and two separate terminal windows.

Open:

1. The `server` folder in one window
2. The `desktop` folder in another window

Each folder should use its own Python environment and be run separately.

### 1. Run the Server

Open a terminal in the `server` folder and create or activate the environment there.

```bash
cd server
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
python app.py
```

The backend runs locally at:

```text
http://127.0.0.1:5001
```

### 2. Run the Desktop App

Open a second terminal in the `desktop` folder and create or activate a different environment there.

```bash
cd desktop
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the desktop application:

```bash
python main.py
```

Important:

- The server must be running before the desktop app can register users or scan attendance.
- The desktop app is currently configured to connect to `http://127.0.0.1:5001`.
- Because the server and desktop have different dependencies, they should be run in different environments.

## Web Portal

The server also provides a browser-based portal for management tasks such as:

- Login
- Dashboard overview
- Attendance viewing
- CSV export
- User management
- Reports
- Settings
- Teacher setup

## Notes

- A default admin account is seeded by the server for development use.
- SQLite is used for local storage.
- SMTP environment variables are needed for password-reset email delivery.
- Face matching uses stored embeddings cached in memory for faster lookup.

## Summary

AFAS combines a desktop attendance client and a Flask-based backend into one integrated attendance platform. The desktop app handles live camera workflows, while the server handles storage, recognition, attendance records, and the management portal.
