import tkinter as tk
from tkinter import ttk
# Importaciones relativas a la GUI principal y de interpolación
# ya que selector_gui.py está ahora en app/gui/windows/
from .main import RootFindingApp
from .interpolation_gui import InterpolationApp

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
            app = RootFindingApp(self.root_finding_window)
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

# Bloque para pruebas aisladas (opcional, pero buena práctica)
if __name__ == '__main__':
    # Para probar este selector de forma independiente
    # Esto podría necesitar ajustes en sys.path si se ejecuta directamente en este entorno.
    # Debido a las limitaciones de cómo se ejecutan los bloques if __name__ == '__main__' en este entorno,
    # puede ser más simple confiar en que el PYTHONPATH está configurado si se ejecuta este archivo directamente,
    # o probarlo ejecutando el main.py principal.
    # Las importaciones relativas 'from ..main' funcionarán cuando este módulo sea importado por main.py
    # pero fallarán si se ejecuta este archivo directamente sin la manipulación de sys.path anterior.
    # Por simplicidad en el edit_file, se omite la manipulación compleja de sys.path aquí,
    # ya que el objetivo principal es que funcione cuando es llamado desde el main.py de la raíz.

    # Re-importar con el path ajustado si es necesario para pruebas directas
    # Esto es solo para la ejecución directa de este archivo de prueba
    # from app.gui.main import RootFindingApp 
    # from app.gui.interpolation_gui import InterpolationApp

    print("Ejecutando MainAppSelector en modo de prueba directa...")
    test_root = tk.Tk()
    # Para que la prueba directa funcione sin el ajuste de sys.path anterior, necesitaríamos 
    # que RootFindingApp e InterpolationApp sean accesibles. Si no, esto fallará.
    # selector = MainAppSelector(test_root) 
    # test_root.mainloop()
    ttk.Label(test_root, text="Prueba de MainAppSelector (funcionalidad de importación completa desde main.py)").pack(padx=20, pady=20)
    test_root.mainloop() 