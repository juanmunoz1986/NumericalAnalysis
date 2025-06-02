import tkinter as tk
# Importar la clase selectora principal desde su nueva ubicación correcta
from app.gui.selector_gui import MainAppSelector

if __name__ == "__main__":
    # Crear la ventana raíz de Tkinter
    root_window = tk.Tk()
    # Instanciar y correr la aplicación selectora
    app = MainAppSelector(root_window)
    # Iniciar el bucle principal de Tkinter para mostrar la GUI
    root_window.mainloop()