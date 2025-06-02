import tkinter as tk
from tkinter import ttk
# Corregir la importación para que coincida con el nombre de clase renombrado en app/gui/main.py
from app.gui.main import RootFindingApp
# Importación para la GUI de interpolación
from app.gui.interpolation_gui import InterpolationApp

class MainAppSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Selector de Herramientas Numéricas")
        self.root.geometry("400x200")

        title_label = ttk.Label(self.root, text="Seleccione una Categoría de Métodos", font=("Arial", 16))
        title_label.pack(pady=20)

        btn_root_finding = ttk.Button(self.root, text="Sistemas de Ecuaciones / Búsqueda de Raíces", command=self.open_root_finding_app)
        btn_root_finding.pack(pady=10, fill='x', padx=50)

        btn_interpolation = ttk.Button(self.root, text="Interpolación y Aproximación", command=self.open_interpolation_app)
        btn_interpolation.pack(pady=10, fill='x', padx=50)

        self.root_finding_window = None
        self.interpolation_window = None

    def open_root_finding_app(self):
        if self.root_finding_window and self.root_finding_window.winfo_exists():
            self.root_finding_window.lift()
        else:
            self.root_finding_window = tk.Toplevel(self.root)
            # La RootFindingApp (antes App) espera la ventana raíz como argumento
            app = RootFindingApp(self.root_finding_window)
            # Protocolo para manejar el cierre de la ventana de RootFindingApp
            self.root_finding_window.protocol("WM_DELETE_WINDOW", self._on_close_root_finding_app)

    def _on_close_root_finding_app(self):
        if self.root_finding_window:
            self.root_finding_window.destroy()
            self.root_finding_window = None

    def open_interpolation_app(self):
        if self.interpolation_window and self.interpolation_window.winfo_exists():
            self.interpolation_window.lift()
        else:
            self.interpolation_window = tk.Toplevel(self.root)
            app_interpolation = InterpolationApp(self.interpolation_window)
            self.interpolation_window.protocol("WM_DELETE_WINDOW", self._on_close_interpolation_app)

    def _on_close_interpolation_app(self):
        if self.interpolation_window:
            self.interpolation_window.destroy()
            self.interpolation_window = None

if __name__ == "__main__":
    main_root = tk.Tk()
    selector_app = MainAppSelector(main_root)
    main_root.mainloop()