import queue
import re
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

from api_client import create_user, upload_sample
from camera_utils import frame_to_dataurl
from ui_theme import COLORS, FONTS, PAD_X, PAD_Y, make_label

PREVIEW_DELAY = 30
REGISTRATION_IMAGE_WIDTH = 320
REGISTRATION_JPEG_QUALITY = 68
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


class RegisterScreen(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS["bg"])
        self.app = app

        canvas = tk.Canvas(self, bg=COLORS["bg"], borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", style="App.Vertical.TScrollbar", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(canvas, bg=COLORS["bg"])
        canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        self.cap = None
        self.running = False
        self.current_frame = None
        self.worker_busy = False
        self.worker_q = queue.Queue()
        self.after_id = None
        self.pending_payload = None
        self.hold_start = None
        self.hold_required = 0.8
        self.ready = False
        self.freeze_instruction = False

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.eye_closed_frames = 0
        self.blink_detected = False
        self.prev_face_center = None
        self.movement_score = 0.0
        self.liveness_passed = False

        shell = tk.Frame(self.inner, bg=COLORS["bg"])
        shell.pack(fill="both", expand=True, padx=32, pady=28)

        header = tk.Frame(shell, bg=COLORS["primary"], highlightbackground=COLORS["primary_dark"], highlightthickness=1)
        header.pack(fill="x", pady=(0, 20))
        make_label(header, "Face Registration", FONTS["title"], fg="#ffffff", bg=COLORS["primary"]).pack(
            anchor="w", padx=PAD_X, pady=(PAD_Y, 4)
        )
        make_label(
            header,
            "Create a profile, capture a clean face sample, and enroll the user for recognition.",
            FONTS["subtitle"],
            fg="#d7f5ef",
            bg=COLORS["primary"]
        ).pack(anchor="w", padx=PAD_X, pady=(0, PAD_Y))

        body = tk.Frame(shell, bg=COLORS["bg"])
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=6)

        form_card = tk.Frame(body, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        make_label(form_card, "Profile Details", FONTS["section"], bg=COLORS["card"]).pack(
            anchor="w", padx=PAD_X, pady=(PAD_Y, 4)
        )
        make_label(
            form_card,
            "Enter the user information first, then start the camera and capture the face sample.",
            FONTS["body"],
            fg=COLORS["muted"],
            bg=COLORS["card"],
            wraplength=360,
            justify="left"
        ).pack(anchor="w", padx=PAD_X, pady=(0, 16))

        form = tk.Frame(form_card, bg=COLORS["card"])
        form.pack(fill="x", padx=PAD_X, pady=(0, 16))
        form.grid_columnconfigure(1, weight=1)

        self.name = self._build_entry(form, 0, "Full Name")
        self.sex = self._build_combo(form, 1, "Sex", ["Male", "Female", "Other"])
        self.role = self._build_combo(form, 2, "Role", ["student", "teacher"])
        self.identifier = self._build_entry(form, 3, "ID Number")
        self.department = self._build_combo(
            form,
            4,
            "Department",
            ["Computer Science", "IT", "Engineering", "Business", "Arts", "Other"]
        )
        self.email = self._build_entry(form, 5, "Email")

        action_row = tk.Frame(form_card, bg=COLORS["card"])
        action_row.pack(fill="x", padx=PAD_X, pady=(0, PAD_Y))
        ttk.Button(action_row, text="Start Registration", style="Primary.TButton", command=self.start_registration).pack(
            side="left"
        )
        ttk.Button(action_row, text="Back to Menu", style="Secondary.TButton", command=self.back).pack(
            side="left", padx=(12, 0)
        )

        camera_card = tk.Frame(body, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        camera_card.grid(row=0, column=1, sticky="nsew")

        top_row = tk.Frame(camera_card, bg=COLORS["panel"])
        top_row.pack(fill="x", padx=PAD_X, pady=(PAD_Y, 8))
        make_label(top_row, "Capture Panel", FONTS["section"], bg=COLORS["panel"]).pack(anchor="w")
        make_label(
            top_row,
            "Keep the face centered inside the guide before capturing.",
            FONTS["body"],
            fg=COLORS["muted"],
            bg=COLORS["panel"]
        ).pack(anchor="w", pady=(4, 0))

        self.loading_frame = tk.Frame(camera_card, bg=COLORS["panel"])
        make_label(self.loading_frame, "Initializing camera...", FONTS["body"], fg=COLORS["muted"], bg=COLORS["panel"]).pack(
            pady=(0, 8)
        )
        self.progress = ttk.Progressbar(
            self.loading_frame,
            style="App.Horizontal.TProgressbar",
            mode="indeterminate",
            length=320
        )
        self.progress.pack()

        preview_shell = tk.Frame(camera_card, bg=COLORS["camera_bg"], highlightbackground=COLORS["border"], highlightthickness=1)
        preview_shell.pack(padx=PAD_X, pady=8)
        preview_shell.pack_propagate(False)
        preview_shell.configure(width=500, height=380)
        self.preview_shell = preview_shell
        self.video_label = tk.Label(
            preview_shell,
            bg=COLORS["camera_bg"],
            width=480,
            height=360,
            bd=0,
            highlightthickness=0,
            fg="#d6d3d1",
            text="Camera preview will appear here\nafter you start registration.",
            font=FONTS["body"],
            justify="center"
        )
        self.video_label.pack(padx=10, pady=10)

        status_card = tk.Frame(camera_card, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        status_card.pack(fill="x", padx=PAD_X, pady=(8, PAD_Y))

        make_label(status_card, "Live Guidance", FONTS["label"], fg=COLORS["muted"], bg=COLORS["card"]).pack(
            anchor="w", padx=18, pady=(14, 4)
        )
        self.instruction = make_label(
            status_card,
            "Fill the form and click Start Registration.",
            FONTS["body"],
            bg=COLORS["card"],
            wraplength=420,
            justify="left"
        )
        self.instruction.pack(anchor="w", padx=18)

        self.status = make_label(
            status_card,
            "Status: Waiting",
            FONTS["status"],
            fg=COLORS["primary"],
            bg=COLORS["card"]
        )
        self.status.pack(anchor="w", padx=18, pady=(10, 12))

        control_row = tk.Frame(camera_card, bg=COLORS["panel"])
        control_row.pack(fill="x", padx=PAD_X, pady=(0, PAD_Y))
        ttk.Button(control_row, text="Capture Face", style="Primary.TButton", command=self.capture_sample).pack(side="left")

        self.after(80, self.process_worker_queue)

    def _build_entry(self, parent, row, label):
        make_label(parent, label, FONTS["label"], fg=COLORS["muted"], bg=COLORS["card"]).grid(
            row=row, column=0, sticky="w", pady=(0, 6)
        )
        entry = tk.Entry(
            parent,
            font=FONTS["body"],
            bd=1,
            relief="solid",
            highlightthickness=0,
            bg="#ffffff",
            fg=COLORS["text"]
        )
        entry.grid(row=row, column=1, sticky="ew", padx=(14, 0), pady=(0, 14), ipady=7)
        return entry

    def _build_combo(self, parent, row, label, values):
        make_label(parent, label, FONTS["label"], fg=COLORS["muted"], bg=COLORS["card"]).grid(
            row=row, column=0, sticky="w", pady=(0, 6)
        )
        combo = ttk.Combobox(parent, values=values, state="readonly", style="App.TCombobox")
        combo.grid(row=row, column=1, sticky="ew", padx=(14, 0), pady=(0, 14))
        combo.current(0)
        return combo

    def reset_liveness(self):
        self.eye_closed_frames = 0
        self.blink_detected = False
        self.prev_face_center = None
        self.movement_score = 0.0
        self.liveness_passed = False

    def reset_all(self):
        self.name.delete(0, tk.END)
        self.identifier.delete(0, tk.END)
        self.email.delete(0, tk.END)
        self.sex.current(0)
        self.role.current(0)
        self.department.current(0)
        self.status.config(text="Status: Waiting", fg=COLORS["primary"])
        self.instruction.config(text="Fill the form and click Start Registration.")
        self.current_frame = None
        self.worker_busy = False
        self.pending_payload = None
        self.hold_start = None
        self.ready = False
        self.freeze_instruction = False
        self.reset_liveness()
        with self.worker_q.mutex:
            self.worker_q.queue.clear()
        self.video_label.config(image="")
        self.video_label.imgtk = None
        self._show_preview_placeholder()

    def on_show(self):
        self.stop_camera()
        self.reset_all()

    def detect_liveness(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            self.prev_face_center = None
            return False

        landmarks = results.multi_face_landmarks[0].landmark
        left_eye_top = landmarks[159]
        left_eye_bottom = landmarks[145]
        eye_dist = abs(left_eye_top.y - left_eye_bottom.y)

        if eye_dist < 0.015:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= 2:
                self.blink_detected = True
            self.eye_closed_frames = 0

        nose = landmarks[1]
        cx = nose.x
        cy = nose.y
        if self.prev_face_center is not None:
            dx = abs(cx - self.prev_face_center[0])
            dy = abs(cy - self.prev_face_center[1])
            self.movement_score += (dx + dy)
        self.prev_face_center = (cx, cy)

        if self.blink_detected or self.movement_score > 0.02:
            self.liveness_passed = True
        return self.liveness_passed

    def start_registration(self):
        if self.worker_busy:
            return

        self.video_label.config(text="", image="")
        self.video_label.imgtk = None
        name = self.name.get().strip()
        sex = self.sex.get()
        role = self.role.get()
        identifier = self.identifier.get().strip()
        department = self.department.get()
        email = self.email.get().strip()

        if not name or not identifier or not email:
            messagebox.showwarning("Missing Fields", "Please fill out all required fields.")
            return
        if not EMAIL_PATTERN.match(email):
            messagebox.showwarning("Invalid Email", "Please enter a valid email address.")
            self.email.focus_set()
            return

        payload = {
            "name": name,
            "sex": sex,
            "role": role,
            "department": department,
            "email": email
        }
        if role == "student":
            payload["student_id"] = identifier
        else:
            payload["teacher_id"] = identifier

        self.pending_payload = payload
        self.hold_start = None
        self.ready = False
        self.freeze_instruction = False
        self.reset_liveness()
        self.status.config(text="Status: Camera starting", fg=COLORS["accent"])
        self.instruction.config(text="Align your face inside the outline.")
        self.start_camera()

    def start_camera(self):
        self.running = True
        self.loading_frame.pack(pady=10)
        self.progress.start(10)
        threading.Thread(target=self.initialize_camera_thread, daemon=True).start()

    def stop_camera(self, clear_preview=True):
        self.running = False
        self.hold_start = None
        self.ready = False
        self.freeze_instruction = False
        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.loading_frame.pack_forget()
        self.progress.stop()
        if clear_preview:
            self.video_label.config(image="")
            self.video_label.imgtk = None
            self._show_preview_placeholder()

    def _show_preview_placeholder(self):
        self.video_label.config(
            image="",
            text="Camera preview will appear here\nafter you start registration.",
            fg="#d6d3d1"
        )

    def initialize_camera_thread(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        warmup_start = time.time()
        while self.running:
            ret, _ = self.cap.read()
            if ret or (time.time() - warmup_start) >= 3:
                self.after(0, self.camera_ready_ui)
                return
            time.sleep(0.01)

    def camera_ready_ui(self):
        if not self.running:
            return
        self.loading_frame.pack_forget()
        self.video_label.config(text="")
        self.update_preview()

    def draw_head_guide(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        head_w = int(w * 0.18)
        head_h = int(h * 0.26)
        head_center = (cx, int(cy * 0.88))
        color = (255, 255, 255)
        thickness = 2
        if self.ready:
            color = (0, 255, 0)
            thickness = 3
        cv2.ellipse(frame, head_center, (head_w, head_h), 0, 0, 360, color, thickness)
        shoulder_y = head_center[1] + head_h + 22
        shoulder_w = int(head_w * 1.25)
        shoulder_h = int(head_h * 0.55)
        cv2.ellipse(frame, (cx, shoulder_y), (shoulder_w, shoulder_h), 0, 200, -20, color, thickness)
        return head_center, head_w, head_h

    def update_preview(self):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            if self.freeze_instruction:
                self.draw_head_guide(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_label.imgtk = img
                self.video_label.configure(image=img)
                self.after_id = self.after(PREVIEW_DELAY, self.update_preview)
                return

            self.ready = False
            now = time.time()
            (gcx, gcy), gw, gh = self.draw_head_guide(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)
            status_text = "Align your face inside the outline."

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_cx = x + w // 2
                face_cy = y + h // 2
                big_enough = w >= 120 and h >= 120
                center_ok = abs(face_cx - gcx) < gw * 0.60 and abs(face_cy - gcy) < gh * 0.70

                if not big_enough:
                    status_text = "Move closer."
                    self.hold_start = None
                    self.reset_liveness()
                elif not center_ok:
                    status_text = "Center your face in the outline."
                    self.hold_start = None
                    self.reset_liveness()
                else:
                    if self.hold_start is None:
                        self.hold_start = now
                    held = now - self.hold_start
                    progress = min(100, int((held / self.hold_required) * 100))
                    if held < self.hold_required:
                        status_text = f"Hold still... {progress}%"
                        self.reset_liveness()
                    else:
                        live = self.detect_liveness(frame)
                        if live:
                            self.ready = True
                            status_text = "Ready. Live face detected."
                        else:
                            status_text = "Waiting for natural movement..."

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if self.ready else (0, 165, 255), 2)
            else:
                self.hold_start = None
                self.reset_liveness()

            self.draw_head_guide(frame)
            self.instruction.config(text=status_text)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)

        self.after_id = self.after(PREVIEW_DELAY, self.update_preview)

    def face_is_large_enough(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return False
        _, _, w, h = max(faces, key=lambda f: f[2] * f[3])
        return w >= 120 and h >= 120

    def capture_sample(self):
        if self.worker_busy or self.current_frame is None:
            return
        if not self.pending_payload:
            messagebox.showwarning("Start First", "Click Start Registration first.")
            return
        if not self.liveness_passed:
            messagebox.showwarning("Live Face Required", "Please look naturally at the camera for a moment.")
            return
        if not self.face_is_large_enough(self.current_frame):
            messagebox.showwarning("Move Closer", "Your face is too far from the camera.")
            return

        image = frame_to_dataurl(
            self.current_frame,
            max_width=REGISTRATION_IMAGE_WIDTH,
            jpeg_quality=REGISTRATION_JPEG_QUALITY
        )
        self.worker_busy = True
        self.freeze_instruction = True
        self.instruction.config(text="Creating user...")
        self.status.config(text="Status: Processing", fg=COLORS["accent"])

        def job():
            try:
                started = time.time()
                print("[REGISTER] Creating user...")
                data = create_user(self.pending_payload)
                print(f"[REGISTER] create_user completed in {time.time() - started:.2f}s")
                new_id = data["user_id"]
                self.worker_q.put(("info", "Uploading face sample..."))
                upload_started = time.time()
                print("[REGISTER] Uploading face sample...")
                up = upload_sample(new_id, "front", image)
                print(f"[REGISTER] upload_sample completed in {time.time() - upload_started:.2f}s")
                print(f"[REGISTER] Total registration request time: {time.time() - started:.2f}s")
                if up.get("duplicate"):
                    self.worker_q.put((
                        "duplicate",
                        {
                            "user_id": up.get("existing_user_id"),
                            "name": up.get("name") or "Existing user",
                            "score": up.get("score", 0.0)
                        }
                    ))
                    return
                if up.get("success"):
                    if data.get("teacher_portal_setup_required"):
                        self.worker_q.put(("teacher_setup", {
                            "message": f"{self.pending_payload.get('name', 'Teacher')} registered successfully.",
                            "url": data.get("setup_url")
                        }))
                    else:
                        self.worker_q.put(("done", f"{self.pending_payload.get('name', 'User')} registered successfully"))
                else:
                    self.worker_q.put(("err", f"Enrollment failed: {up}"))
            except Exception as e:
                self.worker_q.put(("err", f"Error: {e}"))

        threading.Thread(target=job, daemon=True).start()

    def process_worker_queue(self):
        try:
            while True:
                msg, text = self.worker_q.get_nowait()
                if msg == "info":
                    self.instruction.config(text=text)
                elif msg == "duplicate":
                    self.worker_busy = False
                    self.freeze_instruction = False
                    self.status.config(text="Status: Duplicate face detected", fg=COLORS["warning"])
                    info = text
                    score_pct = round(float(info.get("score", 0.0)) * 100, 2)
                    messagebox.showwarning(
                        "Face Already Registered",
                        f"Face already registered.\n\n"
                        f"Matched user: {info.get('name', 'Existing user')}\n"
                        f"User ID: {info.get('user_id', 'N/A')}\n"
                        f"Match score: {score_pct}%\n\n"
                        f"Use the existing account for this person."
                    )
                    self.instruction.config(text="Face already registered. Review the form.")
                elif msg == "done":
                    self.worker_busy = False
                    self.stop_camera()
                    self.status.config(text="Status: Done", fg=COLORS["success"])
                    self.instruction.config(text="Registration complete. Camera closed.")
                    messagebox.showinfo("Done", text)
                    self.back()
                elif msg == "teacher_setup":
                    self.worker_busy = False
                    self.stop_camera()
                    self.status.config(text="Status: Done", fg=COLORS["success"])
                    self.instruction.config(text="Registration complete. Camera closed.")
                    info = text
                    if info.get("url"):
                        messagebox.showinfo(
                            "Teacher Portal Setup",
                            f"{info['message']}\n\nRedirecting you to create the Teacher Portal."
                        )
                        webbrowser.open(info["url"])
                    else:
                        messagebox.showinfo(
                            "Teacher Registered",
                            f"{info['message']}\n\nTeacher portal setup URL was not provided."
                        )
                    self.back()
                elif msg == "err":
                    self.worker_busy = False
                    self.freeze_instruction = False
                    self.status.config(text="Status: Error", fg=COLORS["danger"])
                    messagebox.showerror("Error", text)
        except queue.Empty:
            pass

        self.after(80, self.process_worker_queue)

    def back(self):
        self.stop_camera()
        self.reset_all()
        from home_screen import HomeScreen
        self.app.show(HomeScreen)
