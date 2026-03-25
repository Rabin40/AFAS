import tkinter as tk
from tkinter import ttk


COLORS = {
    "bg": "#f3efe7",
    "panel": "#fbf8f2",
    "panel_alt": "#efe7d8",
    "card": "#fffdf9",
    "border": "#d9ccb9",
    "text": "#1f2933",
    "muted": "#6b7280",
    "primary": "#1e3a8a",
    "primary_dark": "#1a3276",
    "accent": "#d97706",
    "success": "#15803d",
    "warning": "#c2410c",
    "danger": "#b91c1c",
    "camera_bg": "#1c1917",
}

FONTS = {
    "hero": ("Georgia", 28, "bold"),
    "title": ("Georgia", 22, "bold"),
    "subtitle": ("Segoe UI", 12),
    "section": ("Segoe UI", 14, "bold"),
    "body": ("Segoe UI", 11),
    "label": ("Segoe UI", 10, "bold"),
    "status": ("Segoe UI", 10, "bold"),
    "button": ("Segoe UI Semibold", 11),
}

PAD_X = 24
PAD_Y = 18


def apply_theme(app):
    app.configure(bg=COLORS["bg"])

    style = ttk.Style(app)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure(
        "App.TFrame",
        background=COLORS["bg"]
    )
    style.configure(
        "Panel.TFrame",
        background=COLORS["panel"],
        relief="flat"
    )
    style.configure(
        "Card.TFrame",
        background=COLORS["card"],
        relief="flat"
    )
    style.configure(
        "Primary.TButton",
        padding=(18, 12),
        font=FONTS["button"],
        foreground="#ffffff",
        background=COLORS["primary"],
        borderwidth=0
    )
    style.map(
        "Primary.TButton",
        background=[("active", COLORS["primary_dark"])]
    )
    style.configure(
        "Secondary.TButton",
        padding=(18, 12),
        font=FONTS["button"],
        foreground=COLORS["text"],
        background=COLORS["panel_alt"],
        borderwidth=0
    )
    style.map(
        "Secondary.TButton",
        background=[("active", "#e4d8c4")]
    )
    style.configure(
        "App.Vertical.TScrollbar",
        troughcolor=COLORS["bg"],
        background=COLORS["border"],
        arrowcolor=COLORS["text"],
        bordercolor=COLORS["bg"]
    )
    style.configure(
        "App.TCombobox",
        padding=6,
        fieldbackground="#ffffff",
        background="#ffffff"
    )
    style.configure(
        "App.Horizontal.TProgressbar",
        troughcolor=COLORS["panel_alt"],
        background=COLORS["primary"],
        bordercolor=COLORS["panel_alt"],
        lightcolor=COLORS["primary"],
        darkcolor=COLORS["primary"]
    )


def make_label(parent, text, font, fg=None, bg=None, **kwargs):
    return tk.Label(
        parent,
        text=text,
        font=font,
        fg=fg or COLORS["text"],
        bg=bg or parent.cget("bg"),
        **kwargs
    )
