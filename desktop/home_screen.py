import tkinter as tk
from tkinter import ttk

from attendance_screen import AttendanceScreen
from register_screen import RegisterScreen
from ui_theme import COLORS, FONTS, PAD_X, PAD_Y, make_label


class HomeScreen(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLORS["bg"])
        self.app = app

        shell = tk.Frame(self, bg=COLORS["bg"])
        shell.pack(fill="both", expand=True, padx=36, pady=32)

        hero = tk.Frame(shell, bg=COLORS["primary"], highlightbackground=COLORS["primary_dark"], highlightthickness=1)
        hero.pack(fill="x", pady=(0, 22))

        make_label(
            hero,
            "AFAS Desktop",
            FONTS["hero"],
            fg="#ffffff",
            bg=COLORS["primary"]
        ).pack(anchor="w", padx=PAD_X, pady=(PAD_Y, 6))
        make_label(
            hero,
            "A cleaner hub for registration and daily attendance scanning.",
            FONTS["subtitle"],
            fg="#d7f5ef",
            bg=COLORS["primary"]
        ).pack(anchor="w", padx=PAD_X, pady=(0, PAD_Y))

        body = tk.Frame(shell, bg=COLORS["bg"])
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)

        action_card = tk.Frame(
            body,
            bg=COLORS["card"],
            highlightbackground=COLORS["border"],
            highlightthickness=1
        )
        action_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        make_label(action_card, "Choose an Action", FONTS["title"], bg=COLORS["card"]).pack(
            anchor="w", padx=PAD_X, pady=(PAD_Y, 4)
        )
        make_label(
            action_card,
            "Open the flow you need and the system will guide you from there.",
            FONTS["subtitle"],
            fg=COLORS["muted"],
            bg=COLORS["card"]
        ).pack(anchor="w", padx=PAD_X, pady=(0, 16))

        self._action_block(
            action_card,
            "Register Student or Teacher",
            "Create a new profile, validate the face sample, and enroll it for recognition.",
            "Open Registration",
            self.go_register,
            "Primary.TButton"
        ).pack(fill="x", padx=PAD_X, pady=(0, 14))

        self._action_block(
            action_card,
            "Take Attendance",
            "Scan the live camera feed and mark attendance with quick result feedback.",
            "Open Attendance",
            self.go_attendance,
            "Secondary.TButton"
        ).pack(fill="x", padx=PAD_X, pady=(0, 22))

        insight_card = tk.Frame(
            body,
            bg=COLORS["panel_alt"],
            highlightbackground=COLORS["border"],
            highlightthickness=1
        )
        insight_card.grid(row=0, column=1, sticky="nsew")

        make_label(insight_card, "Before You Start", FONTS["section"], bg=COLORS["panel_alt"]).pack(
            anchor="w", padx=PAD_X, pady=(PAD_Y, 10)
        )

        tips = [
            "Keep the camera at eye level for faster face alignment.",
            "Use registration for new users and attendance for returning users.",
            "After a successful attendance scan, ask the person to step away before the next scan."
        ]
        for tip in tips:
            row = tk.Frame(insight_card, bg=COLORS["panel_alt"])
            row.pack(fill="x", padx=PAD_X, pady=6)
            bullet = tk.Canvas(row, width=12, height=12, bg=COLORS["panel_alt"], highlightthickness=0)
            bullet.create_oval(2, 2, 10, 10, fill=COLORS["accent"], outline="")
            bullet.pack(side="left", pady=4)
            make_label(
                row,
                tip,
                FONTS["body"],
                fg=COLORS["text"],
                bg=COLORS["panel_alt"],
                wraplength=280,
                justify="left"
            ).pack(side="left", padx=(10, 0))

        ttk.Button(shell, text="Exit Application", style="Secondary.TButton", command=app.destroy).pack(
            anchor="e", pady=(20, 0)
        )

    def _action_block(self, parent, title, description, button_text, command, style):
        block = tk.Frame(parent, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        make_label(block, title, FONTS["section"], bg=COLORS["card"]).pack(anchor="w", padx=18, pady=(16, 4))
        make_label(
            block,
            description,
            FONTS["body"],
            fg=COLORS["muted"],
            bg=COLORS["card"],
            wraplength=520,
            justify="left"
        ).pack(anchor="w", padx=18, pady=(0, 12))
        ttk.Button(block, text=button_text, style=style, command=command).pack(anchor="w", padx=18, pady=(0, 16))
        return block

    def go_register(self):
        self.app.show(RegisterScreen)

    def go_attendance(self):
        self.app.show(AttendanceScreen)
