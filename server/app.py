import os
import json
import base64
import datetime as dt
import secrets
import random
import smtplib
import warnings
from email.mime.text import MIMEText
from dotenv import load_dotenv

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
from deepface import DeepFace

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response
from zoneinfo import ZoneInfo

from models import db, User, FaceSample, AttendanceLog, PortalUser
from auth import login_required, role_required
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy is deprecated.*",
)

POSES = ["front"]
SAMPLES_PER_POSE = 1

MATCH_THRESHOLD = 0.72
DUPLICATE_THRESHOLD = 0.68
ENROLL_DETECTORS = ["opencv", "mediapipe"]
SCAN_DETECTORS = ["opencv", "mediapipe", "ssd", "retinaface"]
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

EMBEDDING_CACHE = []
PASSWORD_RESET_OTPS = {}


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")
    app.config["SERVER_BOOT_TOKEN"] = secrets.token_hex(16)

    @app.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.before_request
    def invalidate_old_session_after_restart():
        protected_paths = {
            "/dashboard",
            "/attendance",
            "/attendance.csv",
            "/admin/users",
            "/admin/reports",
            "/admin/settings",
            "/admin/cleanup_orphans",
            "/teacher/users",
            "/teacher/reports",
            "/teacher/settings",
        }

        path = request.path or ""
        teacher_shared_routes = {
            "/admin/users": "/teacher/users",
            "/admin/reports": "/teacher/reports",
            "/admin/settings": "/teacher/settings",
        }

        if path.startswith("/teacher/setup/"):
            return

        if session.get("portal_role") == "teacher" and path in teacher_shared_routes:
            return redirect(teacher_shared_routes[path])

        if path in protected_paths:
            if session.get("portal_user_id"):
                if session.get("boot_token") != app.config["SERVER_BOOT_TOKEN"]:
                    session.clear()
                    return redirect(url_for("login"))

    @app.context_processor
    def inject_portal_user():
        display_name = "User"
        portal_role = session.get("portal_role")

        def portal_url(section):
            shared_sections = {"users", "reports", "settings"}
            if section in shared_sections and portal_role == "teacher":
                return f"/teacher/{section}"
            if section in shared_sections:
                return f"/admin/{section}"
            return f"/{section.lstrip('/')}"

        portal_user_id = session.get("portal_user_id")
        if portal_user_id:
            portal_user = PortalUser.query.get(portal_user_id)
            if portal_user:
                if portal_user.role == "teacher" and portal_user.linked_user_id:
                    linked_user = User.query.get(portal_user.linked_user_id)
                    if linked_user:
                        display_name = linked_user.name
                    else:
                        display_name = portal_user.username.capitalize()
                else:
                    display_name = portal_user.username.capitalize()

        return {
            "display_name": display_name,
            "portal_role": portal_role,
            "portal_url": portal_url,
        }

    os.makedirs(app.instance_path, exist_ok=True)
    db_path = os.path.join(app.instance_path, "afas.db")
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + db_path
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    with app.app_context():
        db.create_all()
        seed_portal_users()
        reload_embedding_cache()

    register_routes(app)
    return app


def seed_portal_users():
    if not PortalUser.query.filter_by(username="admin").first():
        db.session.add(
            PortalUser(
                username="admin",
                password=generate_password_hash("admin123"),
                role="admin",
                is_active=True
            )
        )
    db.session.commit()


def validate_user_payload(data):
    role = data.get("role")
    if role not in ("student", "teacher", "admin"):
        return "Invalid role"
    if not data.get("name"):
        return "Name required"
    if not data.get("email"):
        return "Email required"
    if role == "student" and not data.get("student_id"):
        return "Student ID required"
    if role == "teacher" and not data.get("teacher_id"):
        return "Teacher ID required"
    if role == "admin" and not data.get("admin_id"):
        return "Admin ID required"
    return None


def decode_dataurl_jpeg(data_url: str):
    b64 = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image")
    return bgr, img_bytes


def extract_primary_face(bgr_img: np.ndarray, padding_ratio: float = 0.20):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(bgr_img.shape[1], x + w + pad_x)
    y2 = min(bgr_img.shape[0], y + h + pad_y)

    face = bgr_img[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return face


def represent_embedding(bgr_img: np.ndarray, detectors=None):
    cropped_face = extract_primary_face(bgr_img)
    if cropped_face is not None:
        try:
            rep = DeepFace.represent(
                img_path=cropped_face,
                model_name="Facenet",
                detector_backend="skip",
                enforce_detection=False
            )
            if rep and len(rep) > 0:
                return rep[0]["embedding"]
        except Exception as e:
            print(f"[DeepFace] Fast crop path failed: {e}")

    detector_order = detectors or SCAN_DETECTORS

    for det in detector_order:
        try:
            rep = DeepFace.represent(
                img_path=bgr_img,
                model_name="Facenet",
                detector_backend=det,
                enforce_detection=False
            )
            if rep and len(rep) > 0:
                return rep[0]["embedding"]
        except Exception as e:
            print(f"[DeepFace] Detector {det} failed: {e}")

    raise ValueError("No face detected by any detector")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def reload_embedding_cache():
    global EMBEDDING_CACHE
    EMBEDDING_CACHE = []

    samples = FaceSample.query.all()
    for s in samples:
        try:
            embedding = np.array(json.loads(s.embedding_json), dtype=np.float32)
            EMBEDDING_CACHE.append({
                "user_id": s.user_id,
                "embedding": embedding
            })
        except Exception as e:
            print(f"[CACHE] Failed to load sample {s.id}: {e}")

    print(f"[CACHE] Loaded {len(EMBEDDING_CACHE)} face embeddings")


def best_match(embedding: np.ndarray):
    best_user_id = None
    best_score = -1.0

    for item in EMBEDDING_CACHE:
        score = cosine(embedding, item["embedding"])
        if score > best_score:
            best_score = score
            best_user_id = item["user_id"]

    return best_user_id, best_score


def generate_otp():
    return str(random.randint(100000, 999999))


def send_email_otp(to_email: str, otp: str):
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_email or not smtp_password:
        raise RuntimeError("SMTP_EMAIL and SMTP_PASSWORD must be set")

    subject = "AFAS Password Reset OTP"
    body = f"Your AFAS password reset OTP is: {otp}\n\nThis code expires in 5 minutes."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_email
    msg["To"] = to_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.send_message(msg)


def register_routes(app: Flask):

    @app.get("/__whoami")
    def __whoami():
        return jsonify({
            "app": "AFAS server/app.py",
            "port": 5001,
            "has_add_sample": True,
            "has_find_user": True
        })

    @app.get("/admin/cleanup_orphans")
    @role_required("admin")
    def cleanup_orphans():
        deleted_samples = 0
        deleted_logs = 0

        for s in FaceSample.query.all():
            if not User.query.get(s.user_id):
                db.session.delete(s)
                deleted_samples += 1

        for log in AttendanceLog.query.all():
            if not User.query.get(log.user_id):
                db.session.delete(log)
                deleted_logs += 1

        db.session.commit()
        reload_embedding_cache()

        return jsonify({
            "deleted_face_samples": deleted_samples,
            "deleted_attendance_logs": deleted_logs
        })

    @app.get("/")
    def root():
        if session.get("portal_user_id") and session.get("boot_token") == app.config["SERVER_BOOT_TOKEN"]:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.get("/login")
    def login():
        success = request.args.get("success")
        return render_template("login.html", success=success)

    @app.post("/login")
    def login_post():
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        u = PortalUser.query.filter_by(
            username=username,
            is_active=True
        ).first()

        if not u or not check_password_hash(u.password, password):
            return render_template("login.html", error="Invalid credentials")

        session.clear()
        session["portal_user_id"] = u.id
        session["portal_role"] = u.role
        session["boot_token"] = app.config["SERVER_BOOT_TOKEN"]
        return redirect(url_for("dashboard"))

    @app.route("/forgot-password", methods=["GET", "POST"])
    def forgot_password():
        if request.method == "POST":
            username = request.form.get("username", "").strip()

            if not username:
                return render_template("forgot_password.html", error="Username is required")

            portal_user = PortalUser.query.filter_by(username=username, is_active=True).first()
            if not portal_user:
                return render_template("forgot_password.html", error="Username not found")

            if not portal_user.linked_user_id:
                return render_template(
                    "forgot_password.html",
                    error="This account is not linked to a registered user email"
                )

            linked_user = User.query.get(portal_user.linked_user_id)
            if not linked_user or not linked_user.email:
                return render_template(
                    "forgot_password.html",
                    error="No registered email found for this username"
                )

            otp = generate_otp()
            expires_at = dt.datetime.utcnow() + dt.timedelta(minutes=5)

            PASSWORD_RESET_OTPS[username] = {
                "otp": otp,
                "expires_at": expires_at
            }

            try:
                send_email_otp(linked_user.email, otp)
                return render_template(
                    "verify_otp.html",
                    username=username,
                    success=f"OTP sent to {linked_user.email}"
                )
            except Exception as e:
                return render_template(
                    "forgot_password.html",
                    error=f"Failed to send OTP: {e}"
                )

        return render_template("forgot_password.html")

    @app.post("/forgot-password/verify")
    def verify_forgot_password():
        username = request.form.get("username", "").strip()
        otp = request.form.get("otp", "").strip()
        new_password = request.form.get("new_password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        user = PortalUser.query.filter_by(username=username, is_active=True).first()
        if not user:
            return render_template("forgot_password.html", error="Username not found")

        otp_record = PASSWORD_RESET_OTPS.get(username)
        if not otp_record:
            return render_template("verify_otp.html", username=username, error="OTP not found or expired")

        if dt.datetime.utcnow() > otp_record["expires_at"]:
            PASSWORD_RESET_OTPS.pop(username, None)
            return render_template("verify_otp.html", username=username, error="OTP expired")

        if otp != otp_record["otp"]:
            return render_template("verify_otp.html", username=username, error="Invalid OTP")

        if not new_password or not confirm_password:
            return render_template("verify_otp.html", username=username, error="All fields are required")

        if new_password != confirm_password:
            return render_template("verify_otp.html", username=username, error="Passwords do not match")

        user.password = generate_password_hash(new_password)
        db.session.commit()
        PASSWORD_RESET_OTPS.pop(username, None)

        return redirect(url_for("login", success="Password reset successfully. Please log in."))

    @app.route("/teacher/setup/<int:user_id>", methods=["GET", "POST"])
    def teacher_setup(user_id):
        user = User.query.get_or_404(user_id)

        if user.role != "teacher":
            return "Invalid access", 400

        existing_link = PortalUser.query.filter_by(linked_user_id=user.id).first()

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()

            if existing_link:
                return render_template(
                    "teacher_setup.html",
                    user=user,
                    error="Teacher portal account already exists"
                )

            if not username or not password or not confirm_password:
                return render_template(
                    "teacher_setup.html",
                    user=user,
                    error="All fields are required"
                )

            if password != confirm_password:
                return render_template(
                    "teacher_setup.html",
                    user=user,
                    error="Passwords do not match"
                )

            username_exists = PortalUser.query.filter_by(username=username).first()
            if username_exists:
                return render_template(
                    "teacher_setup.html",
                    user=user,
                    error="Username already taken"
                )

            portal_user = PortalUser(
                username=username,
                password=generate_password_hash(password),
                role="teacher",
                is_active=True,
                linked_user_id=user.id
            )
            db.session.add(portal_user)
            db.session.commit()

            return redirect(url_for("login", success="Teacher portal created successfully. Please log in."))

        return render_template("teacher_setup.html", user=user)

    @app.get("/dashboard")
    @login_required
    def dashboard():
        page = max(request.args.get("page", default=1, type=int), 1)
        per_page = 20
        total_users = User.query.count()
        total_students = User.query.filter_by(role="student").count()
        total_attendance = AttendanceLog.query.count()

        today = dt.date.today().isoformat()
        today_present = AttendanceLog.query.filter_by(day=today).count()

        q = (
            db.session.query(AttendanceLog, User)
            .join(User, User.id == AttendanceLog.user_id)
            .order_by(AttendanceLog.timestamp.desc())
        )
        total_pages = max((total_attendance + per_page - 1) // per_page, 1)
        if page > total_pages:
            page = total_pages

        rows = q.offset((page - 1) * per_page).limit(per_page).all()

        attendance = []
        for log, user in rows:
            ny_tz = ZoneInfo("America/New_York")
            ts = log.timestamp
            ts_utc = ts.replace(tzinfo=dt.timezone.utc)
            ts_ny = ts_utc.astimezone(ny_tz)

            attendance.append({
                "name": user.name,
                "department": user.department or "",
                "date": ts_ny.strftime("%Y-%m-%d"),
                "time": ts_ny.strftime("%I:%M %p"),
            })

        return render_template(
            "dashboard.html",
            total_users=total_users,
            total_students=total_students,
            total_attendance=total_attendance,
            today=today,
            today_present=today_present,
            attendance=attendance,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            role=session.get("portal_role"),
            active="dashboard",
        )

    @app.get("/attendance")
    @login_required
    def attendance():
        day = request.args.get("day", dt.date.today().isoformat())
        page = max(request.args.get("page", default=1, type=int), 1)
        per_page = 20

        q = (
            db.session.query(AttendanceLog, User)
            .join(User, User.id == AttendanceLog.user_id)
            .filter(AttendanceLog.day == day)
            .order_by(AttendanceLog.timestamp.asc())
        )
        total_rows = q.count()
        total_pages = max((total_rows + per_page - 1) // per_page, 1)
        if page > total_pages:
            page = total_pages

        rows = q.offset((page - 1) * per_page).limit(per_page).all()

        return render_template(
            "attendance.html",
            rows=rows,
            day=day,
            page=page,
            per_page=per_page,
            total_rows=total_rows,
            total_pages=total_pages,
            active="attendance",
        )

    @app.get("/attendance.csv")
    @login_required
    def attendance_csv():
        day = request.args.get("day", dt.date.today().isoformat())
        q = db.session.query(AttendanceLog, User).join(User, User.id == AttendanceLog.user_id)
        q = q.filter(AttendanceLog.day == day).order_by(User.name.asc())
        rows = q.all()

        def gen():
            yield "day,name,role,identifier,email,score,timestamp\n"
            for log, user in rows:
                identifier = user.identifier() or ""
                yield f"{log.day},{user.name},{user.role},{identifier},{user.email},{log.score},{log.timestamp}\n"

        return Response(gen(), mimetype="text/csv")

    @app.get("/admin/users")
    @app.get("/teacher/users")
    @role_required("admin", "teacher")
    def admin_users():
        role_q = request.args.get("role")
        page = max(request.args.get("page", default=1, type=int), 1)
        per_page = 20
        q = User.query
        if role_q:
            q = q.filter_by(role=role_q)

        q = q.order_by(User.created_at.desc())
        total_rows = q.count()
        total_pages = max((total_rows + per_page - 1) // per_page, 1)
        if page > total_pages:
            page = total_pages

        rows = q.offset((page - 1) * per_page).limit(per_page).all()
        return render_template(
            "user_management.html",
            users=rows,
            page=page,
            per_page=per_page,
            total_rows=total_rows,
            total_pages=total_pages,
            role_q=role_q,
            active="users",
            role=session.get("portal_role")
        )

    @app.post("/admin/users/<int:user_id>/toggle")
    @role_required("admin")
    def toggle_user(user_id):
        user = User.query.get_or_404(user_id)
        user.is_active = not user.is_active
        db.session.commit()
        return redirect(request.referrer or url_for("admin_users"))

    @app.post("/admin/users/<int:user_id>/delete")
    @role_required("admin")
    def delete_user(user_id):
        user = User.query.get_or_404(user_id)

        FaceSample.query.filter_by(user_id=user_id).delete()
        AttendanceLog.query.filter_by(user_id=user_id).delete()

        linked_portal = PortalUser.query.filter_by(linked_user_id=user_id).first()
        if linked_portal:
            db.session.delete(linked_portal)

        db.session.delete(user)
        db.session.commit()
        reload_embedding_cache()

        return redirect(request.referrer or url_for("admin_users"))

    @app.get("/admin/reports")
    @app.get("/teacher/reports")
    @login_required
    def admin_reports():
        import calendar

        users = User.query.filter_by(is_active=True).all()
        total_users = len(users)

        today = dt.date.today().isoformat()
        today_logs = AttendanceLog.query.filter_by(day=today).all()
        today_present_count = len({log.user_id for log in today_logs})

        total_present = round((today_present_count / total_users) * 100, 1) if total_users else 0
        total_absent = round(100 - total_present, 1) if total_users else 0

        dept_map = {}
        for user in users:
            dept = user.department or "Unknown"
            if dept not in dept_map:
                dept_map[dept] = {"total": 0, "present": 0}
            dept_map[dept]["total"] += 1

        for log in today_logs:
            user = User.query.get(log.user_id)
            if user and user.is_active:
                dept = user.department or "Unknown"
                if dept in dept_map:
                    dept_map[dept]["present"] += 1

        dept_labels = []
        dept_values = []

        for dept, stats in dept_map.items():
            dept_labels.append(dept)
            percent = round((stats["present"] / stats["total"]) * 100, 1) if stats["total"] else 0
            dept_values.append(percent)

        avg_department_present = round(sum(dept_values) / len(dept_values), 1) if dept_values else 0

        month_labels = []
        month_values = []
        time_labels = []
        time_values = []

        now = dt.date.today()
        logs = AttendanceLog.query.all()
        ny_tz = ZoneInfo("America/New_York")

        for i in range(5, -1, -1):
            year = now.year
            month = now.month - i

            while month <= 0:
                month += 12
                year -= 1

            month_name = calendar.month_abbr[month]
            month_labels.append(month_name)

            monthly_logs = [
                log for log in logs
                if log.timestamp.year == year and log.timestamp.month == month
            ]

            unique_users_present = len({log.user_id for log in monthly_logs})
            percent = round((unique_users_present / total_users) * 100, 1) if total_users else 0
            month_values.append(percent)

        hourly_counts = {hour: 0 for hour in range(24)}
        for log in logs:
            ts = log.timestamp
            ts_utc = ts.replace(tzinfo=dt.timezone.utc)
            ts_ny = ts_utc.astimezone(ny_tz)
            hourly_counts[ts_ny.hour] += 1

        for hour in range(24):
            time_labels.append(dt.time(hour=hour).strftime("%I %p"))
            time_values.append(hourly_counts[hour])

        total = len(dept_labels)
        start = 1 if total > 0 else 0
        end = total

        return render_template(
            "reports.html",
            total_present=total_present,
            total_absent=total_absent,
            avg_department_present=avg_department_present,
            dept_labels=dept_labels,
            dept_values=dept_values,
            month_labels=month_labels,
            month_values=month_values,
            time_labels=time_labels,
            time_values=time_values,
            start=start,
            end=end,
            total=total,
            active="reports",
            role=session.get("portal_role"),
        )

    @app.route("/admin/settings", methods=["GET", "POST"])
    @app.route("/teacher/settings", methods=["GET", "POST"])
    @login_required
    def admin_settings():
        user_id = session.get("portal_user_id")
        user = PortalUser.query.get(user_id)

        if request.method == "POST":
            current_password = request.form.get("current_password")
            new_password = request.form.get("new_password")
            confirm_password = request.form.get("confirm_password")

            if not check_password_hash(user.password, current_password):
                return render_template(
                    "settings.html",
                    error="Current password is incorrect",
                    user=user,
                    active="settings"
                )

            if new_password != confirm_password:
                return render_template(
                    "settings.html",
                    error="Passwords do not match",
                    user=user,
                    active="settings"
                )

            user.password = generate_password_hash(new_password)
            db.session.commit()

            return render_template(
                "settings.html",
                success="Password updated successfully",
                user=user,
                active="settings"
            )

        return render_template("settings.html", user=user, active="settings")

    @app.post("/api/users/create")
    def api_users_create():
        data = request.get_json(force=True)

        error = validate_user_payload(data)
        if error:
            return jsonify({"success": False, "error": error}), 400

        role = data["role"]

        user = User(
            name=data["name"].strip(),
            sex=data.get("sex"),
            role=role,
            department=data.get("department"),
            email=data["email"].strip(),
            is_active=True,
            student_id=data.get("student_id") if role == "student" else None,
            teacher_id=data.get("teacher_id") if role == "teacher" else None,
            admin_id=data.get("admin_id") if role == "admin" else None,
        )

        db.session.add(user)
        db.session.commit()

        if user.role == "teacher":
            return jsonify({
                "success": True,
                "user_id": user.id,
                "teacher_portal_setup_required": True,
                "teacher_id": user.teacher_id,
                "name": user.name,
                "setup_url": f"http://127.0.0.1:5001/teacher/setup/{user.id}"
            })

        return jsonify({"success": True, "user_id": user.id})

    @app.post("/api/enroll/find_user")
    def api_enroll_find_user():
        data = request.get_json(force=True)
        bgr, _ = decode_dataurl_jpeg(data["image"])

        try:
            emb = represent_embedding(bgr, detectors=ENROLL_DETECTORS)
        except Exception:
            return jsonify({"success": False, "error": "No face detected"}), 200

        new_emb = np.array(emb, dtype=np.float32)

        best_user_id, best_score = best_match(new_emb)

        if best_user_id is not None and best_score >= DUPLICATE_THRESHOLD:
            u = User.query.get(best_user_id)
            return jsonify({
                "success": True,
                "match": True,
                "user_id": best_user_id,
                "name": u.name if u else "",
                "score": float(best_score)
            }), 200

        return jsonify({
            "success": True,
            "match": False,
            "score": float(best_score)
        }), 200

    @app.post("/api/enroll/add_sample")
    def api_enroll_add_sample():
        data = request.get_json(force=True)
        user_id = int(data["user_id"])
        pose = data["pose"]

        if pose not in POSES:
            return jsonify({"success": False, "error": "Invalid pose"}), 400

        user = User.query.get(user_id)
        if not user or not user.is_active:
            return jsonify({"success": False, "error": "User not found/active"}), 404

        bgr, jpeg_bytes = decode_dataurl_jpeg(data["image"])

        try:
            emb = represent_embedding(bgr, detectors=ENROLL_DETECTORS)
        except Exception:
            return jsonify({"success": False, "error": "No face detected"}), 200

        new_emb = np.array(emb, dtype=np.float32)

        for item in EMBEDDING_CACHE:
            score = cosine(new_emb, item["embedding"])
            if score >= DUPLICATE_THRESHOLD and item["user_id"] != user_id:
                return jsonify({
                    "success": False,
                    "duplicate": True,
                    "existing_user_id": item["user_id"],
                    "score": float(score),
                }), 200

        sample = FaceSample(
            user_id=user_id,
            pose=pose,
            embedding_json=json.dumps(emb),
            image_jpeg=jpeg_bytes,
        )
        db.session.add(sample)
        db.session.commit()

        EMBEDDING_CACHE.append({
            "user_id": user_id,
            "embedding": new_emb
        })

        count_pose = FaceSample.query.filter_by(user_id=user_id, pose=pose).count()
        return jsonify({"success": True, "count_for_pose": count_pose})

    @app.post("/api/attendance/scan_and_mark")
    def api_attendance_scan_and_mark():
        try:
            data = request.get_json(force=True)
            bgr, _ = decode_dataurl_jpeg(data["image"])

            emb = np.array(represent_embedding(bgr), dtype=np.float32)
            user_id, score = best_match(emb)

            if user_id is None or score < MATCH_THRESHOLD:
                return jsonify({"found": False, "message": "User not registered", "score": float(score)})

            user = User.query.get(user_id)
            if not user or not user.is_active:
                return jsonify({"found": False, "message": "User not registered", "score": float(score)})

            day = dt.date.today().isoformat()
            already = AttendanceLog.query.filter_by(user_id=user_id, day=day).first()

            already_marked = True
            if not already:
                already_marked = False
                log = AttendanceLog(user_id=user_id, day=day, status="present", score=float(score))
                db.session.add(log)
                db.session.commit()

            return jsonify({
                "found": True,
                "name": user.name,
                "user_id": user.id,
                "score": float(score),
                "already_marked": already_marked,
                "day": day,
            })

        except Exception as e:
            print("[scan_and_mark] ERROR:", repr(e))
            return jsonify({"found": False, "message": "Server error"}), 500

    @app.get("/logout")
    def logout():
        session.clear()
        resp = redirect(url_for("login"))
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp


app = create_app()

if __name__ == "__main__":
    try:
        from waitress import serve

        serve(app, host="127.0.0.1", port=5001)
    except ImportError:
        app.run(debug=False, port=5001)
