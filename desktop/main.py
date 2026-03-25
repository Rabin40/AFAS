import tkinter as tk
from home_screen import HomeScreen
from register_screen import RegisterScreen
from attendance_screen import AttendanceScreen
from ui_theme import apply_theme


class AFASApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AFAS Desktop")
        self.geometry("1180x760")
        self.resizable(False, False)
        apply_theme(self)

        self.container = tk.Frame(self, bg=self["bg"])
        self.container.pack(fill="both", expand=True)

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {
            HomeScreen: HomeScreen(self.container, self),
            RegisterScreen: RegisterScreen(self.container, self),
            AttendanceScreen: AttendanceScreen(self.container, self)
        }

        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

        self.current_screen = None
        self.show(HomeScreen)

    def show(self, screen):
        print(f"[MAIN] Switching to screen: {screen.__name__}")

        # clean up current screen before switching
        if self.current_screen is not None:
            current_frame = self.frames[self.current_screen]

            if hasattr(current_frame, "stop_camera"):
                try:
                    current_frame.stop_camera()
                except Exception as e:
                    print(f"[MAIN] stop_camera error on {self.current_screen.__name__}: {e}")

            elif hasattr(current_frame, "on_hide"):
                try:
                    current_frame.on_hide()
                except Exception as e:
                    print(f"[MAIN] on_hide error on {self.current_screen.__name__}: {e}")

            elif hasattr(current_frame, "running"):
                current_frame.running = False

        frame = self.frames[screen]
        frame.tkraise()
        self.current_screen = screen

        if hasattr(frame, "on_show"):
            frame.after(50, frame.on_show)


if __name__ == "__main__":
    AFASApp().mainloop()
