from app.gui import app
import tkinter as tk


if __name__ == "__main__":
    root = tk.Tk()
    application = app.App(root)
    root.mainloop()
