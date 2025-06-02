import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np

# Importar los módulos de backend de interpolación
from app.methods.interpolation import finite_differences
from app.methods.interpolation import newton_divided_differences

class InterpolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Métodos de Interpolación y Aproximación")
        self.root.geometry("450x250") # Ajustar tamaño según necesidad

        title_label = ttk.Label(self.root, text="Seleccione un Método", font=("Arial", 16))
        title_label.pack(pady=15)

        btn_finite_diff = ttk.Button(self.root, text="Aproximación de Derivadas (Dif. Finitas)", command=self.open_finite_differences_window)
        btn_finite_diff.pack(pady=10, fill='x', padx=50)

        btn_newton_div_diff = ttk.Button(self.root, text="Polinomio de Newton (Dif. Divididas)", command=self.open_newton_divided_differences_window)
        btn_newton_div_diff.pack(pady=10, fill='x', padx=50)

        # Variables para manejar las ventanas Toplevel
        self.finite_diff_window = None
        self.newton_dd_window = None

    def open_finite_differences_window(self):
        if self.finite_diff_window and self.finite_diff_window.winfo_exists():
            self.finite_diff_window.lift()
            return

        self.finite_diff_window = tk.Toplevel(self.root)
        self.finite_diff_window.title("Diferencias Finitas para Derivadas")
        self.finite_diff_window.geometry("700x500")
        self.finite_diff_window.protocol("WM_DELETE_WINDOW", lambda: self._close_window_generic(self.finite_diff_window, 'finite_diff_window'))

        # Contenido de la ventana de Diferencias Finitas
        ttk.Label(self.finite_diff_window, text="Valores de x (separados por comas):").pack(pady=(10,0))
        self.fd_x_entries = ttk.Entry(self.finite_diff_window, width=80)
        self.fd_x_entries.pack(pady=5, padx=10)
        self.fd_x_entries.insert(0, "1, 2, 3, 4, 5") # Ejemplo

        ttk.Label(self.finite_diff_window, text="Valores de y (f(x), separados por comas):").pack(pady=(10,0))
        self.fd_y_entries = ttk.Entry(self.finite_diff_window, width=80)
        self.fd_y_entries.pack(pady=5, padx=10)
        self.fd_y_entries.insert(0, "1, 4, 9, 16, 25") # Ejemplo y=x^2

        btn_calculate_fd = ttk.Button(self.finite_diff_window, text="Calcular Derivadas", command=self.solve_finite_differences)
        btn_calculate_fd.pack(pady=10)

        ttk.Label(self.finite_diff_window, text="Resultados:").pack(pady=(10,0))
        self.fd_results_text = scrolledtext.ScrolledText(self.finite_diff_window, height=15, width=80)
        self.fd_results_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.fd_results_text.config(state=tk.DISABLED)

    def solve_finite_differences(self):
        self.fd_results_text.config(state=tk.NORMAL)
        self.fd_results_text.delete(1.0, tk.END)
        try:
            x_str = self.fd_x_entries.get().split(',')
            y_str = self.fd_y_entries.get().split(',')
            
            if len(x_str) != len(y_str):
                raise ValueError("Las listas de x e y deben tener la misma cantidad de elementos.")
            if len(x_str) < 2:
                raise ValueError("Se necesitan al menos dos puntos para calcular diferencias.")

            x_vals = np.array([float(val.strip()) for val in x_str])
            y_vals = np.array([float(val.strip()) for val in y_str])

            if not finite_differences.is_spacing_equal(x_vals):
                self.fd_results_text.insert(tk.END, "Error: Los valores de x deben estar igualmente espaciados para este método.\n")
                self.fd_results_text.config(state=tk.DISABLED)
                return

            resultados = finite_differences.numerical_derivatives_with_formulas(x_vals, y_vals)
            
            output = "Aproximaciones de Derivadas:\n\n"
            for r in resultados:
                output += f"Punto x = {r['x']}\n"
                if r.get('sucesiva'):
                    output += f"  Sucesiva:  {r['sucesiva'][1]:.6f} (Fórmula: {r['sucesiva'][0]})\n"
                if r.get('central'):
                    output += f"  Central:   {r['central'][1]:.6f} (Fórmula: {r['central'][0]})\n"
                if r.get('regresiva'):
                    output += f"  Regresiva: {r['regresiva'][1]:.6f} (Fórmula: {r['regresiva'][0]})\n"
                output += "-\n"
            self.fd_results_text.insert(tk.END, output)

        except ValueError as ve:
            self.fd_results_text.insert(tk.END, f"Error de valor: {str(ve)}\nRecuerde ingresar números separados por comas.")
        except Exception as e:
            self.fd_results_text.insert(tk.END, f"Ocurrió un error: {str(e)}\n")
        finally:
            self.fd_results_text.config(state=tk.DISABLED)

    def open_newton_divided_differences_window(self):
        if self.newton_dd_window and self.newton_dd_window.winfo_exists():
            self.newton_dd_window.lift()
            return

        self.newton_dd_window = tk.Toplevel(self.root)
        self.newton_dd_window.title("Polinomio de Newton (Diferencias Divididas)")
        self.newton_dd_window.geometry("800x650") # Ajustar tamaño
        self.newton_dd_window.protocol("WM_DELETE_WINDOW", lambda: self._close_window_generic(self.newton_dd_window, 'newton_dd_window'))

        # Contenido de la ventana de Newton
        input_frame = ttk.Frame(self.newton_dd_window)
        input_frame.pack(pady=10, padx=10, fill='x')

        ttk.Label(input_frame, text="Valores de x (separados por comas):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.ndd_x_entries = ttk.Entry(input_frame, width=60)
        self.ndd_x_entries.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        self.ndd_x_entries.insert(0, "0, 0.5, 1, 1.5, 2") # Ejemplo

        ttk.Label(input_frame, text="Valores de y (f(x), separados por comas):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.ndd_y_entries = ttk.Entry(input_frame, width=60)
        self.ndd_y_entries.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        self.ndd_y_entries.insert(0, "0, 2.5, 5.8, 8.5, 10.2") # Ejemplo
        
        ttk.Label(input_frame, text="Valor de x para interpolar:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.ndd_x_eval_entry = ttk.Entry(input_frame, width=15)
        self.ndd_x_eval_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.ndd_x_eval_entry.insert(0, "1.25") # Ejemplo

        input_frame.columnconfigure(1, weight=1)

        btn_calculate_ndd = ttk.Button(self.newton_dd_window, text="Calcular e Interpolar", command=self.solve_newton_divided_differences)
        btn_calculate_ndd.pack(pady=10)

        ttk.Label(self.newton_dd_window, text="Resultados:").pack(pady=(10,0))
        self.ndd_results_text = scrolledtext.ScrolledText(self.newton_dd_window, height=25, width=90)
        self.ndd_results_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.ndd_results_text.config(state=tk.DISABLED)

    def solve_newton_divided_differences(self):
        self.ndd_results_text.config(state=tk.NORMAL)
        self.ndd_results_text.delete(1.0, tk.END)
        try:
            x_str = self.ndd_x_entries.get().split(',')
            y_str = self.ndd_y_entries.get().split(',')
            x_eval_str = self.ndd_x_eval_entry.get()

            if len(x_str) != len(y_str):
                raise ValueError("Las listas de x e y deben tener la misma cantidad de elementos.")
            if len(x_str) < 1:
                raise ValueError("Se necesita al menos un punto.")
            if not x_eval_str.strip():
                raise ValueError("Debe ingresar un valor de x para interpolar.")

            x_vals = np.array([float(val.strip()) for val in x_str])
            y_vals = np.array([float(val.strip()) for val in y_str])
            x_to_evaluate = float(x_eval_str.strip())

            value_table, formula_table = newton_divided_differences.build_divided_difference_table(x_vals, y_vals)
            coefficients = newton_divided_differences.get_newton_coefficients(value_table)
            y_interpolated = newton_divided_differences.evaluate_newton_polynomial(x_vals, coefficients, x_to_evaluate)

            output = "Tabla de Diferencias Divididas (Valores):\n"
            # Formatear la tabla (simplificado para GUI)
            n_table = len(value_table)
            headers = ["x_i", "y_i"] + [f"D{j}" for j in range(1, n_table)]
            col_widths = [max(8, len(h)) for h in headers] 

            header_line = " | ".join([f"{headers[j]:^{col_widths[j]}}" for j in range(min(len(headers), n_table))])
            output += header_line + "\n"
            output += "-" * len(header_line) + "\n"

            for i in range(n_table):
                row_str = f"{x_vals[i]:<{col_widths[0]}.4g} | "
                for j in range(n_table - i):
                    val = value_table[i][j]
                    if isinstance(val, (float, np.float64)):
                        row_str += f"{val:<{max(8,col_widths[j+1] if j+1 < len(col_widths) else 8)}.4f} | "
                    else:
                         row_str += f"{'':<{max(8,col_widths[j+1] if j+1 < len(col_widths) else 8)}} | " # Si no hay valor (celdas vacías)
                output += row_str.strip().strip('|').strip() + "\n"
            output += "\n"
            
            output += f"Coeficientes del Polinomio de Newton (a0, a1, ...):\n{np.array2string(np.array(coefficients), precision=6, suppress_small=True)}\n\n"
            output += f"Polinomio P(x) = {coefficients[0]:.4f}"
            for i in range(1, len(coefficients)):
                term = f" + {coefficients[i]:.4f}"
                for k in range(i):
                    term += f"(x - {x_vals[k]})"
                output += term
            output += "\n\n"
            output += f"Valor interpolado en x = {x_to_evaluate}: P({x_to_evaluate}) = {y_interpolated:.6f}\n"

            self.ndd_results_text.insert(tk.END, output)

        except ValueError as ve:
            self.ndd_results_text.insert(tk.END, f"Error de valor: {str(ve)}\nRecuerde ingresar números separados por comas y un valor válido para interpolar.")
        except Exception as e:
            self.ndd_results_text.insert(tk.END, f"Ocurrió un error: {str(e)}\n")
        finally:
            self.ndd_results_text.config(state=tk.DISABLED)

    def _close_window_generic(self, window, window_var_name_str):
        # Reutilizar esta función si es necesario para cerrar Toplevels
        window_attr = getattr(self, window_var_name_str, None)
        if window_attr:
            window_attr.destroy()
            setattr(self, window_var_name_str, None)


if __name__ == '__main__':
    # Este bloque es para probar InterpolationApp de forma aislada si es necesario
    # No se ejecutará cuando se importe desde el main.py principal
    test_root = tk.Tk()
    app = InterpolationApp(test_root)
    test_root.mainloop() 