import datetime as dt
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(150), nullable=False)
    sex = db.Column(db.String(10))
    role = db.Column(db.String(20), nullable=False)  # student/teacher/admin
    department = db.Column(db.String(100))

    student_id = db.Column(db.String(50), unique=True)
    teacher_id = db.Column(db.String(50), unique=True)
    admin_id = db.Column(db.String(50), unique=True)

    email = db.Column(db.String(255), unique=True, nullable=False)

    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)

    def identifier(self):
        if self.role == "student":
            return self.student_id
        if self.role == "teacher":
            return self.teacher_id
        if self.role == "admin":
            return self.admin_id
        return None


class FaceSample(db.Model):
    __tablename__ = "face_samples"
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    pose = db.Column(db.String(20), nullable=False)
    embedding_json = db.Column(db.Text, nullable=False)
    image_jpeg = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)


class AttendanceLog(db.Model):
    __tablename__ = "attendance_logs"
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    day = db.Column(db.String(10), nullable=False, index=True)  # YYYY-MM-DD
    timestamp = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)

    status = db.Column(db.String(50), nullable=False, default="present")
    score = db.Column(db.Float, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("user_id", "day", name="uq_attendance_user_day"),
    )


class PortalUser(db.Model):
    __tablename__ = "portal_users"
    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  # demo only
    role = db.Column(db.String(20), nullable=False)       # admin/teacher
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    # link portal account to actual registered user
    linked_user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        unique=True,
        nullable=True
    )

    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)