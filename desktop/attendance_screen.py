import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from api_client import scan_attendance
from camera_utils import frame_to_dataurl
from ui_theme import COLORS, FONTS, PAD_X, PAD_Y, make_label

PREVIEW_DELAY = 30
ATTENDANCE_INTERVAL = 0.25
FACE_HOLD_SECONDS = 0.35
FACE_HOLD_GRACE_SECONDS = 0.30
ATTENDANCE_IMAGE_WIDTH = 320
ATTENDANCE_JPEG_QUALITY = 65
RESULT_HOLD_SECONDS = 2.0
SUCCESS_COOLDOWN_SECONDS = 4.0
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class AttendanceScreen(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS["bg"])
        self.app = app

        self.cap = None
        self.running = False
        self.after_id = None
        self.camera_ready_called = False
        self.loop_started = False
        self.screen_active = False
        self.last_scan = 0.0
        self.last_result = "Scanning..."
        self.result_color = COLORS["primary"]
        self.result_bgr = (255, 0, 0)
        self.scan_in_progress = False
        self.server_cooldown_until = 0.0
        self.result_hold_until = 0.0
        self.ui_q = queue.Queue()
        self.face_hold_start = None
        self.face_hold_grace_until = 0.0
        self.require_face_clear = False

        shell = tk.Frame(self, bg=COLORS["bg"])
        shell.pack(fill="both", expand=True, padx=36, pady=28)

        header = tk.Frame(shell, bg=COLORS["primary"], highlightbackground=COLORS["primary_dark"], highlightthickness=1)
        header.pack(fill="x", pady=(0, 20))
        make_label(header, "Attendance Scanner", FONTS["title"], fg="#ffffff", bg=COLORS["primary"]).pack(
            anchor="w", padx=PAD_X, pady=(PAD_Y, 4)
        )
        make_label(
            header,
            "Fast, guided scanning with clearer result feedback for each attendance check.",
            FONTS["subtitle"],
            fg="#d7f5ef",
            bg=COLORS["primary"]
        ).pack(anchor="w", padx=PAD_X, pady=(0, PAD_Y))

        body = tk.Frame(shell, bg=COLORS["bg"])
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=7)
        body.grid_columnconfigure(1, weight=4)

        scanner_card = tk.Frame(body, bg=COLORS["panel"], highlightbackground=COLORS["border"], highlightthickness=1)
        scanner_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        top = tk.Frame(scanner_card, bg=COLORS["panel"])
        top.pack(fill="x", padx=PAD_X, pady=(PAD_Y, 10))
        make_label(top, "Live Camera", FONTS["section"], bg=COLORS["panel"]).pack(anchor="w")
        make_label(
            top,
            "Ask the next person to center their face, hold briefly, and then step away after the result appears.",
            FONTS["body"],
            fg=COLORS["muted"],
            bg=COLORS["panel"],
            wraplength=560,
            justify="left"
        ).pack(anchor="w", pady=(4, 0))

        self.loading_frame = tk.Frame(scanner_card, bg=COLORS["panel"])
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
        self.progress.start(10)

        preview_shell = tk.Frame(scanner_card, bg=COLORS["camera_bg"], highlightbackground=COLORS["border"], highlightthickness=1)
        preview_shell.pack(padx=PAD_X, pady=8)
        preview_shell.pack_propagate(False)
        preview_shell.configure(width=660, height=500)
        self.preview_shell = preview_shell
        self.video_label = tk.Label(
            preview_shell,
            bg=COLORS["camera_bg"],
            width=640,
            height=480,
            bd=0,
            highlightthickness=0,
            fg="#d6d3d1",
            text="Camera preview will appear here\nwhen attendance scanning starts.",
            font=FONTS["body"],
            justify="center"
        )
        self.video_label.pack(padx=10, pady=10)

        side_card = tk.Frame(body, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        side_card.grid(row=0, column=1, sticky="nsew")

        make_label(side_card, "Scan Status", FONTS["section"], bg=COLORS["card"]).pack(
            anchor="w", padx=PAD_X, pady=(PAD_Y, 6)
        )
        make_label(
            side_card,
            "The scanner will show the latest result here and wait for the face to clear before scanning again.",
            FONTS["body"],
            fg=COLORS["muted"],
            bg=COLORS["card"],
            wraplength=280,
            justify="left"
        ).pack(anchor="w", padx=PAD_X, pady=(0, 14))

        self.result_panel = tk.Frame(side_card, bg=COLORS["panel_alt"], highlightbackground=COLORS["border"], highlightthickness=1)
        self.result_panel.pack(fill="x", padx=PAD_X, pady=(0, 18))
        make_label(self.result_panel, "Current Result", FONTS["label"], fg=COLORS["muted"], bg=COLORS["panel_alt"]).pack(
            anchor="w", padx=18, pady=(14, 4)
        )
        self.info = make_label(
            self.result_panel,
            "Status: Waiting...",
            FONTS["section"],
            fg=COLORS["primary"],
            bg=COLORS["panel_alt"],
            wraplength=250,
            justify="left"
        )
        self.info.pack(anchor="w", padx=18, pady=(0, 14))

        tips_card = tk.Frame(side_card, bg=COLORS["card"])
        tips_card.pack(fill="x", padx=PAD_X, pady=(0, 18))
        make_label(tips_card, "Quick Tips", FONTS["label"], fg=COLORS["muted"], bg=COLORS["card"]).pack(anchor="w")
        for text in [
            "If a result says already marked, ask the person to step away.",
            "A centered face gets scanned faster than a face near the edges.",
            "The scanner pauses briefly after a successful match to avoid duplicate triggers."
        ]:
            make_label(
                tips_card,
                f"- {text}",
                FONTS["body"],
                bg=COLORS["card"],
                wraplength=280,
                justify="left"
            ).pack(anchor="w", pady=4)

        ttk.Button(side_card, text="Back to Menu", style="Secondary.TButton", command=self.back).pack(
            anchor="w", padx=PAD_X, pady=(0, PAD_Y)
        )

        self.after(60, self._process_ui_queue)

    def reset_liveness(self):
        self.face_hold_start = None
        self.face_hold_grace_until = 0.0

    def draw_scan_guide(self, frame, ready=False):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        guide_w = int(w * 0.20)
        guide_h = int(h * 0.28)
        guide_center = (cx, int(cy * 0.92))

        color = (0, 255, 0) if ready else (255, 255, 255)
        thickness = 3 if ready else 2

        cv2.ellipse(frame, guide_center, (guide_w, guide_h), 0, 0, 360, color, thickness)

        shoulder_y = guide_center[1] + guide_h + 24
        shoulder_w = int(guide_w * 1.3)
        shoulder_h = int(guide_h * 0.58)
        cv2.ellipse(frame, (cx, shoulder_y), (shoulder_w, shoulder_h), 0, 200, -20, color, thickness)

        return guide_center, guide_w, guide_h

    def face_ready_for_scan(self, frame, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if len(faces) == 0:
            if now > self.face_hold_grace_until:
                self.face_hold_start = None
            self.require_face_clear = False
            return False, "Show your face to scan", None

        h, w = frame.shape[:2]
        (gcx, gcy), gw, gh = self.draw_scan_guide(frame, ready=False)
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_cx = x + fw // 2
        face_cy = y + fh // 2
        centered = abs(face_cx - gcx) < (gw * 0.92) and abs(face_cy - gcy) < (gh * 0.98)
        large_enough = fw >= 100 and fh >= 100

        if not large_enough:
            if now > self.face_hold_grace_until:
                self.face_hold_start = None
            return False, "Move closer", (x, y, fw, fh, False)
        if not centered:
            if now > self.face_hold_grace_until:
                self.face_hold_start = None
            return False, "Center your face", (x, y, fw, fh, False)

        if self.face_hold_start is None:
            self.face_hold_start = now
        self.face_hold_grace_until = now + FACE_HOLD_GRACE_SECONDS
        held = now - self.face_hold_start
        if held < FACE_HOLD_SECONDS:
            progress = min(100, int((held / FACE_HOLD_SECONDS) * 100))
            return False, f"Hold still... {progress}%", (x, y, fw, fh, False)
        return True, "Scanning...", (x, y, fw, fh, True)

    def on_show(self):
        print("[ATTENDANCE] on_show called")
        if self.screen_active:
            return

        self.screen_active = True
        self.running = True
        self.loop_started = False
        self.camera_ready_called = False
        self.after_id = None
        self.last_result = "Scanning..."
        self.result_color = COLORS["primary"]
        self.result_bgr = (255, 0, 0)
        self.scan_in_progress = False
        self.last_scan = 0.0
        self.server_cooldown_until = 0.0
        self.result_hold_until = 0.0
        self.require_face_clear = False
        self.reset_liveness()

        self.info.config(text="Status: Initializing camera...", fg=COLORS["primary"])
        self.loading_frame.pack(pady=20)
        self.video_label.config(text="", image="")
        self.video_label.imgtk = None
        threading.Thread(target=self.initialize_camera_thread, daemon=True).start()

    def initialize_camera_thread(self):
        print("[ATTENDANCE] Opening camera...")
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
        if not self.running or self.camera_ready_called:
            return
        self.camera_ready_called = True
        self.loading_frame.pack_forget()
        self.video_label.config(text="")
        self.info.config(text="Status: Scanning...", fg=COLORS["primary"])
        if not self.loop_started:
            self.loop_started = True
            self.update_loop()

    def _show_preview_placeholder(self):
        self.video_label.config(
            image="",
            text="Camera preview will appear here\nwhen attendance scanning starts.",
            fg="#d6d3d1"
        )
        self.video_label.imgtk = None

    def _process_ui_queue(self):
        try:
            while True:
                msg = self.ui_q.get_nowait()
                if not msg:
                    continue
                kind, payload = msg
                if kind == "result":
                    self.last_result = payload["text"]
                    self.result_color = payload["fg"]
                    self.result_bgr = payload["bgr"]
                    self.server_cooldown_until = max(
                        self.server_cooldown_until,
                        time.time() + payload.get("cooldown", 0.0)
                    )
                    if payload.get("require_face_clear"):
                        self.require_face_clear = True
                    self.result_hold_until = time.time() + RESULT_HOLD_SECONDS
                elif kind == "server_down":
                    self.last_result = payload
                    self.result_color = COLORS["danger"]
                    self.result_bgr = (0, 0, 255)
                    self.server_cooldown_until = time.time() + 3.0
                    self.result_hold_until = time.time() + RESULT_HOLD_SECONDS
                elif kind == "scan_done":
                    self.scan_in_progress = False
        except queue.Empty:
            pass

        self.after(60, self._process_ui_queue)

    def update_loop(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            now = time.time()
            self.draw_scan_guide(frame, ready=False)
            ready_to_scan, status_text, face_box = self.face_ready_for_scan(frame, now)
            showing_result = now < self.result_hold_until

            if not showing_result and not ready_to_scan and not self.scan_in_progress:
                self.last_result = status_text
                self.result_color = COLORS["primary"]
                self.result_bgr = (255, 0, 0)

            if face_box is not None:
                x, y, fw, fh, box_ready = face_box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + fw, y + fh),
                    (0, 255, 0) if box_ready else (0, 165, 255),
                    2
                )

            if ready_to_scan:
                self.draw_scan_guide(frame, ready=True)

            can_scan = (
                (now - self.last_scan) > ATTENDANCE_INTERVAL
                and not self.scan_in_progress
                and now >= self.server_cooldown_until
                and not showing_result
                and not self.require_face_clear
                and ready_to_scan
            )

            if can_scan:
                self.last_scan = now
                self.scan_in_progress = True
                self.last_result = "Scanning..."
                self.result_color = COLORS["primary"]
                self.result_bgr = (255, 0, 0)
                image_dataurl = frame_to_dataurl(
                    frame,
                    max_width=ATTENDANCE_IMAGE_WIDTH,
                    jpeg_quality=ATTENDANCE_JPEG_QUALITY
                )

                def job():
                    try:
                        print("[ATTENDANCE] Sending frame to backend...")
                        data = scan_attendance(image_dataurl)
                        print("[ATTENDANCE] Response:", data)
                        if data.get("found"):
                            name = data.get("name", "Unknown")
                            already = data.get("already_marked", False)
                            text = f"{name} (already marked)" if already else f"{name} (marked present)"
                            self.ui_q.put(("result", {
                                "text": text,
                                "fg": COLORS["warning"] if already else COLORS["success"],
                                "bgr": (0, 165, 255) if already else (0, 255, 0),
                                "cooldown": SUCCESS_COOLDOWN_SECONDS,
                                "require_face_clear": True
                            }))
                        else:
                            self.ui_q.put(("result", {
                                "text": data.get("message", "Not found"),
                                "fg": COLORS["danger"],
                                "bgr": (0, 0, 255),
                                "cooldown": 1.0
                            }))
                    except Exception as e:
                        print("[ATTENDANCE] Error:", e)
                        self.ui_q.put(("server_down", "Server offline / crashed"))
                    finally:
                        self.ui_q.put(("scan_done", True))

                threading.Thread(target=job, daemon=True).start()

            cv2.putText(
                frame,
                self.last_result,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                self.result_bgr,
                2
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.imgtk = img
            self.video_label.configure(image=img)
            self.info.config(text=f"Status: {self.last_result}", fg=self.result_color)

        self.after_id = self.after(PREVIEW_DELAY, self.update_loop)

    def back(self):
        print("[ATTENDANCE] Back pressed")
        from home_screen import HomeScreen
        self.app.show(HomeScreen)

    def stop_camera(self):
        print("[ATTENDANCE] stop_camera called")
        self.running = False
        self.screen_active = False
        self.loop_started = False
        self.camera_ready_called = False
        self.scan_in_progress = False
        self.reset_liveness()
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
        self._show_preview_placeholder()
        self.info.config(text="Status: Waiting...", fg=COLORS["primary"])
