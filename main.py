from app.gui import main as main_window
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = main_window.App(root)
    root.mainloop()