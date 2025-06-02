import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np

# Importar los módulos de backend de interpolación
from app.methods.interpolation import finite_differences
from app.methods.interpolation import newton_divided_differences
from app.methods.interpolation import data_resampling

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
        self.finite_diff_window.geometry("700x650") # Ajustado para más campos
        self.finite_diff_window.protocol("WM_DELETE_WINDOW", lambda: self._close_window_generic(self.finite_diff_window, 'finite_diff_window'))

        # --- Entradas para el rango de x ---
        input_frame_x_range = ttk.Frame(self.finite_diff_window)
        input_frame_x_range.pack(pady=(10,0), padx=10, fill='x')

        ttk.Label(input_frame_x_range, text="Valor Inicial de x (x₀):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.fd_x_initial_entry = ttk.Entry(input_frame_x_range, width=15)
        self.fd_x_initial_entry.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        self.fd_x_initial_entry.insert(0, "1")

        ttk.Label(input_frame_x_range, text="Valor Final de x (x₁):").grid(row=0, column=2, padx=5, pady=2, sticky='w')
        self.fd_x_final_entry = ttk.Entry(input_frame_x_range, width=15)
        self.fd_x_final_entry.grid(row=0, column=3, padx=5, pady=2, sticky='w')
        self.fd_x_final_entry.insert(0, "5")

        ttk.Label(input_frame_x_range, text="Paso Actual (h):").grid(row=0, column=4, padx=5, pady=2, sticky='w')
        self.fd_h_entry = ttk.Entry(input_frame_x_range, width=15)
        self.fd_h_entry.grid(row=0, column=5, padx=5, pady=2, sticky='w')
        self.fd_h_entry.insert(0, "1")
        
        # --- Entrada para los valores de y (datos originales) ---
        ttk.Label(self.finite_diff_window, text="Valores de y (f(x) originales, separados por comas):").pack(pady=(10,0))
        self.fd_y_entries = ttk.Entry(self.finite_diff_window, width=80)
        self.fd_y_entries.pack(pady=5, padx=10)
        self.fd_y_entries.insert(0, "1, 4, 9, 16, 25") # Ejemplo y=x^2

        # --- Sección de Remuestreo ---
        ttk.Separator(self.finite_diff_window, orient='horizontal').pack(fill='x', pady=10, padx=10)

        resample_frame = ttk.Frame(self.finite_diff_window)
        resample_frame.pack(pady=(5,0), padx=10, fill='x')
        
        ttk.Label(resample_frame, text="Nuevo Paso (h') para Remuestreo:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.fd_h_new_entry = ttk.Entry(resample_frame, width=15)
        self.fd_h_new_entry.grid(row=0, column=1, padx=5, pady=2, sticky='w')
        self.fd_h_new_entry.insert(0, "0.5") # Ejemplo

        btn_resample = ttk.Button(resample_frame, text="Remuestrear Datos y Actualizar Campos", command=self.resample_and_update_fd_inputs)
        btn_resample.grid(row=0, column=2, padx=10, pady=5, sticky='e')
        
        resample_frame.columnconfigure(2, weight=1) # Para que el botón se alinee a la derecha

        ttk.Separator(self.finite_diff_window, orient='horizontal').pack(fill='x', pady=10, padx=10)
        # --- Fin Sección de Remuestreo ---


        btn_calculate_fd = ttk.Button(self.finite_diff_window, text="Calcular Derivadas (con datos actuales/remuestreados)", command=self.solve_finite_differences)
        btn_calculate_fd.pack(pady=10)

        ttk.Label(self.finite_diff_window, text="Resultados:").pack(pady=(10,0))
        self.fd_results_text = scrolledtext.ScrolledText(self.finite_diff_window, height=15, width=80)
        self.fd_results_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.fd_results_text.config(state=tk.DISABLED)

    def resample_and_update_fd_inputs(self):
        self.fd_results_text.config(state=tk.NORMAL)
        self.fd_results_text.delete(1.0, tk.END)
        try:
            # 1. Obtener datos originales de la GUI
            x_initial_str = self.fd_x_initial_entry.get()
            x_final_str = self.fd_x_final_entry.get()
            h_current_str = self.fd_h_entry.get() # Paso actual de los datos y
            y_original_str_list = self.fd_y_entries.get().split(',')
            h_new_str = self.fd_h_new_entry.get()

            if not all([x_initial_str, x_final_str, h_current_str, self.fd_y_entries.get(), h_new_str]):
                raise ValueError("Todos los campos para el remuestreo deben estar completos (x₀, x₁, h actual, y originales, h nuevo).")

            x_initial = float(x_initial_str)
            x_final = float(x_final_str)
            h_current = float(h_current_str)
            h_new = float(h_new_str)

            if h_current <= 0:
                raise ValueError("El 'Paso Actual (h)' debe ser positivo.")
            if h_new <= 0:
                raise ValueError("El 'Nuevo Paso (h') para Remuestreo' debe ser positivo.")
            if x_initial > x_final: # Esta validación ya existe en solve_finite_differences pero es bueno tenerla aquí
                 raise ValueError("El 'Valor Inicial de x' no puede ser mayor que el 'Valor Final de x'.")

            # 2. Generar x_original_vals basados en los parámetros de x y el h_current
            # Esta lógica es similar a la de solve_finite_differences para generar los x que corresponden a los y entrados
            if x_initial == x_final and h_current > 0: # Un solo punto, si h_current se da como positivo, es ambiguo.
                 num_points_orig = 1
            elif x_initial == x_final: # Un solo punto
                 num_points_orig = 1
            else:
                num_points_orig = int(round((x_final - x_initial) / h_current)) + 1
            
            x_original_vals = np.linspace(x_initial, x_final, num_points_orig)
            
            if len(x_original_vals) != len(y_original_str_list):
                raise ValueError(f"La cantidad de 'Valores de y originales' ({len(y_original_str_list)}) no coincide con "
                                 f"la cantidad de puntos x generados por x₀, x₁ y 'Paso Actual (h)' ({len(x_original_vals)}).")

            y_original_vals = np.array([float(val.strip()) for val in y_original_str_list])

            if len(x_original_vals) < 2 : # Newton necesita al menos 2 puntos para un polinomio de grado 1.
                                        # O 1 punto si solo se quiere "evaluar" en ese mismo punto.
                                        # El backend de Newton maneja la cantidad de puntos, pero una verificación temprana es útil.
                raise ValueError("Se necesitan al menos dos puntos originales (x,y) para la interpolación de remuestreo.")


            # 3. Llamar al backend de re-muestreo
            # El rango para los nuevos puntos x será el mismo x_initial y x_final
            self.fd_results_text.insert(tk.END, f"Remuestreando datos con h'={h_new} en el rango [{x_initial}, {x_final}]...\\n")
            
            x_resampled, y_resampled = data_resampling.resample_data_newton(
                x_original_vals, 
                y_original_vals,
                x_initial, # x_target_start
                x_final,   # x_target_end
                h_new
            )

            # 4. Actualizar los campos de la GUI
            self.fd_x_initial_entry.delete(0, tk.END)
            self.fd_x_initial_entry.insert(0, f"{x_resampled[0]:.10g}") # Usar .10g para buena precisión

            self.fd_x_final_entry.delete(0, tk.END)
            self.fd_x_final_entry.insert(0, f"{x_resampled[-1]:.10g}")

            self.fd_h_entry.delete(0, tk.END)
            # Calcular el h real de los datos remuestreados si hay más de un punto
            actual_new_h = h_new
            if len(x_resampled) > 1:
                actual_new_h = x_resampled[1] - x_resampled[0]
            self.fd_h_entry.insert(0, f"{actual_new_h:.10g}")


            y_resampled_str = ", ".join([f"{yval:.10g}" for yval in y_resampled])
            self.fd_y_entries.delete(0, tk.END)
            self.fd_y_entries.insert(0, y_resampled_str)

            self.fd_results_text.insert(tk.END, "¡Campos actualizados con los datos remuestreados!\\n")
            self.fd_results_text.insert(tk.END, f"  Nuevos x generados ({len(x_resampled)} puntos): de {x_resampled[0]:.4f} a {x_resampled[-1]:.4f} con paso aprox. {actual_new_h:.4f}\\n")
            self.fd_results_text.insert(tk.END, "  Nuevos y generados (primeros 5): " + ", ".join([f"{y:.4f}" for y in y_resampled[:5]]) + ("..." if len(y_resampled) > 5 else "") + "\\n")
            self.fd_results_text.insert(tk.END, "Ahora puede presionar 'Calcular Derivadas'.\\n")

        except ValueError as ve:
            self.fd_results_text.insert(tk.END, f"Error de valor: {str(ve)}\\n")
        except Exception as e:
            self.fd_results_text.insert(tk.END, f"Ocurrió un error inesperado durante el remuestreo: {str(e)}\\n")
        finally:
            self.fd_results_text.config(state=tk.DISABLED)

    def solve_finite_differences(self):
        self.fd_results_text.config(state=tk.NORMAL)
        self.fd_results_text.delete(1.0, tk.END)
        try:
            x_initial_str = self.fd_x_initial_entry.get()
            x_final_str = self.fd_x_final_entry.get()
            h_str = self.fd_h_entry.get()
            y_str_list = self.fd_y_entries.get().split(',')

            if not all([x_initial_str, x_final_str, h_str]):
                raise ValueError("Los campos de 'Valor Inicial de x', 'Valor Final de x' y 'Paso (h)' no pueden estar vacíos.")

            x_initial = float(x_initial_str)
            x_final = float(x_final_str)
            h = float(h_str)

            if h <= 0:
                raise ValueError("El paso (h) debe ser un número positivo.")
            if x_initial > x_final:
                raise ValueError("El 'Valor Inicial de x' no puede ser mayor que el 'Valor Final de x'.")
            
            # Generar los valores de x usando np.arange.
            # np.arange puede tener problemas de precisión con flotantes para el punto final.
            # Una forma más robusta es calcular el número de puntos.
            num_points = int(round((x_final - x_initial) / h)) + 1
            x_vals = np.linspace(x_initial, x_final, num_points)
            
            if len(x_vals) != len(y_str_list):
                raise ValueError(f"La cantidad de valores 'y' ({len(y_str_list)}) no coincide con la cantidad de puntos 'x' generados ({len(x_vals)}) para el rango y paso especificados.")
            
            y_vals = np.array([float(val.strip()) for val in y_str_list])

            if len(x_vals) < 2:
                raise ValueError("Se necesitan al menos dos puntos (generados por el rango y paso) para calcular diferencias.")

            # La comprobación is_spacing_equal puede ser redundante si generamos x_vals con un paso constante h,
            # pero el backend podría depender de ello o realizar sus propias comprobaciones de 'h'.
            # Por ahora, lo dejamos, pero podría ser un punto a revisar si el backend se adapta.
            if not finite_differences.is_spacing_equal(x_vals, tolerance=1e-9): # Añadimos una tolerancia
                self.fd_results_text.insert(tk.END, "Advertencia: Los valores de x generados no parecen estar perfectamente espaciados según la función de comprobación. Esto podría ser un problema de precisión o del backend.\\n")
                # Considerar si detener o continuar. Por ahora, continuamos.

            resultados = finite_differences.numerical_derivatives_with_formulas(x_vals, y_vals)
            
            output = "Aproximaciones de Derivadas:\\n"
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