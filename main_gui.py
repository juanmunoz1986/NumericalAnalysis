import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2Tk
from app.methods import fac_LU, jacobi_method, nw_ry_no_lineales_relajacion as newton_nl


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto de Métodos Numéricos")
        self.root.geometry("450x420") # Más alto para el nuevo botón

        title_label = ttk.Label(self.root, text="Seleccione un Método Numérico", font=("Arial", 16))
        title_label.pack(pady=15)

        btn_lu = ttk.Button(self.root, text="Factorización LU (Sistemas Lineales)", command=self.open_lu_window)
        btn_lu.pack(pady=5, fill='x', padx=50) # Menos pady

        btn_sor = ttk.Button(self.root, text="Gauss-Seidel / SOR (Sist. Lineales)", command=self.open_sor_window)
        btn_sor.pack(pady=5, fill='x', padx=50)

        btn_jacobi = ttk.Button(self.root, text="Jacobi Clásico (Sist. Lineales)", command=self.open_jacobi_window)
        btn_jacobi.pack(pady=5, fill='x', padx=50)

        btn_newton_nl = ttk.Button(self.root, text="Newton-Raphson (Sist. No Lineales)", command=self.open_newton_nl_window)
        btn_newton_nl.pack(pady=5, fill='x', padx=50)
        
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(pady=10, fill='x', padx=20)
        
        btn_compare = ttk.Button(self.root, text="Comparar Métodos Lineales", command=self.open_compare_linear_window)
        btn_compare.pack(pady=10, fill='x', padx=50)

        self.lu_matrix_size = 0 # Para almacenar el tamaño actual de la matriz LU
        self.entries_A = []
        self.entries_b = []
        self.results_lu_text = None # Para el widget de texto de resultados LU
        self.lu_window = None # Para la ventana Toplevel de LU
        
        # Atributos para la ventana SOR
        self.sor_matrix_size = 0
        self.entries_A_sor = []
        self.entries_b_sor = []
        self.entries_x0_sor = [] 
        self.entry_w_sor = None
        self.entry_tol_sor = None
        self.entry_max_iter_sor = None
        self.results_sor_text = None
        self.sor_window = None
        # Un solo frame para contener A, b, x0
        self.sor_input_area_frame = None 

        # Atributos para la ventana Jacobi
        self.jacobi_matrix_size = 0
        self.entries_A_jacobi = []
        self.entries_b_jacobi = []
        self.entries_x0_jacobi = [] 
        self.entry_tol_jacobi = None
        self.entry_max_iter_jacobi = None
        self.results_jacobi_text = None
        self.jacobi_window = None
        # Un solo frame para contener A, b, x0
        self.jacobi_input_area_frame = None

        # Atributos para la ventana de comparación
        self.compare_matrix_size = 0
        self.entries_A_comp = []; self.entries_b_comp = []; self.entries_x0_comp = []
        self.entry_tol_comp = None; self.entry_max_iter_comp = None; self.entry_w_comp = None
        self.compare_window = None
        self.compare_input_area_frame = None
        self.compare_notebook = None 
        self.compare_results_text_lu = None
        self.compare_results_text_jacobi = None
        self.compare_results_text_sor = None
        # Nuevo para graficación
        self.compare_plot_frame = None 
        self.compare_plot_canvas = None # Para mantener referencia al canvas

    def open_lu_window(self):
        if self.lu_window and self.lu_window.winfo_exists():
            self.lu_window.lift()
            return

        self.lu_window = tk.Toplevel(self.root)
        self.lu_window.title("Factorización LU")
        self.lu_window.geometry("700x600") # Ajustar tamaño según necesidad
        self.lu_window.protocol("WM_DELETE_WINDOW", lambda: self._close_window_generic(self.lu_window, 'lu_window'))

        # Frame para la configuración del tamaño
        size_config_frame = ttk.Frame(self.lu_window)
        size_config_frame.pack(pady=10)

        ttk.Label(size_config_frame, text="Tamaño de la matriz (N x N): ").pack(side=tk.LEFT, padx=5)
        self.lu_matrix_size_entry = ttk.Entry(size_config_frame, width=5)
        self.lu_matrix_size_entry.pack(side=tk.LEFT, padx=5)
        self.lu_matrix_size_entry.insert(0, "3") # Valor por defecto

        btn_generate_matrix = ttk.Button(size_config_frame, text="Generar Matriz", command=self._generate_lu_matrix_entries)
        btn_generate_matrix.pack(side=tk.LEFT, padx=5)

        # Frame para las entradas de la matriz y el vector (se llenará dinámicamente)
        self.lu_matrix_entry_frame = ttk.Frame(self.lu_window)
        self.lu_matrix_entry_frame.pack(pady=10)
        
        # Botón para resolver
        btn_solve_lu = ttk.Button(self.lu_window, text="Resolver por LU", command=self.solve_lu)
        btn_solve_lu.pack(pady=10)

        # Área de resultados
        ttk.Label(self.lu_window, text="Resultados:").pack(pady=5)
        self.results_lu_text = tk.Text(self.lu_window, height=20, width=80)
        self.results_lu_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        self.results_lu_text.config(state=tk.DISABLED)

        self._generate_lu_matrix_entries() # Generar la matriz inicial con el tamaño por defecto

    def _close_window_generic(self, window, window_var_name_str):
        window = getattr(self, window_var_name_str, None)
        if window:
            window.destroy()
            setattr(self, window_var_name_str, None)

    def _generate_lu_matrix_entries(self):
        try:
            n = int(self.lu_matrix_size_entry.get())
            if not (2 <= n <= 10): # Limitar tamaño por practicidad de la GUI
                if self.results_lu_text:
                    self.results_lu_text.config(state=tk.NORMAL)
                    self.results_lu_text.delete(1.0, tk.END)
                    self.results_lu_text.insert(tk.END, "Error: El tamaño de la matriz debe estar entre 2 y 10.\n")
                    self.results_lu_text.config(state=tk.DISABLED)
                return
            self.lu_matrix_size = n
        except ValueError:
            if self.results_lu_text:
                self.results_lu_text.config(state=tk.NORMAL)
                self.results_lu_text.delete(1.0, tk.END)
                self.results_lu_text.insert(tk.END, "Error: El tamaño de la matriz debe ser un número entero.\n")
                self.results_lu_text.config(state=tk.DISABLED)
            return

        # Limpiar frame anterior
        for widget in self.lu_matrix_entry_frame.winfo_children():
            widget.destroy()

        self.entries_A = []
        self.entries_b = []

        # Crear nuevas entradas para la Matriz A
        matrix_a_frame = ttk.LabelFrame(self.lu_matrix_entry_frame, text="Matriz A")
        matrix_a_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor='n')
        for i in range(self.lu_matrix_size):
            row_entries = []
            for j in range(self.lu_matrix_size):
                entry = ttk.Entry(matrix_a_frame, width=5)
                entry.grid(row=i, column=j, padx=2, pady=2)
                row_entries.append(entry)
            self.entries_A.append(row_entries)

        # Crear nuevas entradas para el Vector b
        vector_b_frame = ttk.LabelFrame(self.lu_matrix_entry_frame, text="Vector b")
        vector_b_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor='n')
        for i in range(self.lu_matrix_size):
            entry = ttk.Entry(vector_b_frame, width=5)
            entry.grid(row=i, column=0, padx=2, pady=2)
            self.entries_b.append(entry)
        
        if self.results_lu_text:
            self.results_lu_text.config(state=tk.NORMAL)
            self.results_lu_text.delete(1.0, tk.END) # Limpiar resultados anteriores
            self.results_lu_text.config(state=tk.DISABLED)

    def solve_lu(self):
        if not self.lu_window or not self.lu_window.winfo_exists() or self.lu_matrix_size == 0:
            # No hacer nada si la ventana no existe o la matriz no ha sido generada
            return 
            
        self.results_lu_text.config(state=tk.NORMAL)
        self.results_lu_text.delete(1.0, tk.END)
        
        try:
            # Leer Matriz A
            matrix_A_list = []
            for i in range(self.lu_matrix_size):
                row_vals = []
                for j in range(self.lu_matrix_size):
                    val = self.entries_A[i][j].get()
                    if not val:
                        self.results_lu_text.insert(tk.END, f"Error: El valor A[{i+1}][{j+1}] no puede estar vacío.\n")
                        self.results_lu_text.config(state=tk.DISABLED)
                        return
                    row_vals.append(float(val))
                matrix_A_list.append(row_vals)
            
            # Leer Vector b
            vector_b_list = []
            for i in range(self.lu_matrix_size):
                val = self.entries_b[i].get() # entries_b es ahora una lista plana
                if not val:
                    self.results_lu_text.insert(tk.END, f"Error: El valor b[{i+1}] no puede estar vacío.\n")
                    self.results_lu_text.config(state=tk.DISABLED)
                    return
                vector_b_list.append(float(val))

            A_np = np.array(matrix_A_list, dtype=float)
            b_np = np.array(vector_b_list, dtype=float)

            resultado = fac_LU.resolver_sistema_lu(A_np, b_np)

            if isinstance(resultado, str):
                self.results_lu_text.insert(tk.END, resultado + "\n")
            else:
                P, L, U, x = resultado
                self.results_lu_text.insert(tk.END, "Factorización LU Exitosa:\n\n")
                self.results_lu_text.insert(tk.END, "Matriz de Permutación P:\n")
                self.results_lu_text.insert(tk.END, np.array2string(P, precision=4, suppress_small=True) + "\n\n")
                self.results_lu_text.insert(tk.END, "Matriz Triangular Inferior L:\n")
                self.results_lu_text.insert(tk.END, np.array2string(L, precision=4, suppress_small=True) + "\n\n")
                self.results_lu_text.insert(tk.END, "Matriz Triangular Superior U:\n")
                self.results_lu_text.insert(tk.END, np.array2string(U, precision=4, suppress_small=True) + "\n\n")
                self.results_lu_text.insert(tk.END, "Vector Solución x:\n")
                self.results_lu_text.insert(tk.END, np.array2string(x, precision=10, suppress_small=True) + "\n")

        except ValueError:
            self.results_lu_text.insert(tk.END, "Error: Todos los valores de la matriz A y el vector b deben ser números válidos.\n")
        except IndexError: # Puede ocurrir si entries_A o entries_b no se llenaron correctamente
             self.results_lu_text.insert(tk.END, "Error: Discrepancia en el tamaño de la matriz. Intente regenerar la matriz.\n")
        except Exception as e:
            self.results_lu_text.insert(tk.END, f"Ocurrió un error inesperado en la GUI: {str(e)}\n")
        
        self.results_lu_text.config(state=tk.DISABLED)

    def open_newton_nl_window(self):
        if hasattr(self, 'newton_nl_window') and self.newton_nl_window and self.newton_nl_window.winfo_exists():
            self.newton_nl_window.lift()
            return

        self.newton_nl_window = tk.Toplevel(self.root)
        self.newton_nl_window.title("Newton-Raphson (Sistemas No Lineales)")
        self.newton_nl_window.geometry("850x750") # Ajustar tamaño para pestañas
        self.newton_nl_window.protocol("WM_DELETE_WINDOW", lambda: self._close_window_generic(self.newton_nl_window, 'newton_nl_window'))

        # Crear el Notebook (pestañas)
        self.newton_nl_notebook = ttk.Notebook(self.newton_nl_window)
        self.newton_nl_notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # --- Pestaña 1: Configuración y Parámetros ---
        config_params_frame = ttk.Frame(self.newton_nl_notebook, padding=10)
        self.newton_nl_notebook.add(config_params_frame, text='Configuración y Parámetros')
        
        # Frame para seleccionar tipo de sistema (dentro de la pestaña 1)
        system_frame = ttk.LabelFrame(config_params_frame, text="Tipo de Sistema No Lineal")
        system_frame.pack(pady=10, padx=10, fill='x')
        
        self.newton_nl_system_var = tk.StringVar(value="sistema_2x2")
        ttk.Radiobutton(system_frame, text="Sistema 2x2 Estándar (predefinido)", 
                        variable=self.newton_nl_system_var, value="sistema_2x2", command=self._toggle_newton_nl_input_mode).pack(anchor='w', padx=10, pady=2)
        ttk.Radiobutton(system_frame, text="Sistema 2x2 Personalizado (ingresar texto)", 
                        variable=self.newton_nl_system_var, value="personalizado_2x2", command=self._toggle_newton_nl_input_mode).pack(anchor='w', padx=10, pady=2)
        
        # Frame para la definición de funciones (personalizado) (dentro de la pestaña 1)
        self.newton_custom_functions_frame = ttk.LabelFrame(config_params_frame, text="Definición de Funciones (Sistema Personalizado)")
        self.newton_custom_functions_frame.pack(pady=10, padx=10, fill='x')

        eq_entry_frame = ttk.Frame(self.newton_custom_functions_frame)
        eq_entry_frame.pack(fill='x', pady=5)
        ttk.Label(eq_entry_frame, text="F1(x,y) =").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.text_f1_eq = tk.Text(eq_entry_frame, height=2, width=60)
        self.text_f1_eq.grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(eq_entry_frame, text="F2(x,y) =").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.text_f2_eq = tk.Text(eq_entry_frame, height=2, width=60)
        self.text_f2_eq.grid(row=1, column=1, padx=5, pady=2, sticky='ew')
        eq_entry_frame.columnconfigure(1, weight=1)

        jac_entry_frame = ttk.Frame(self.newton_custom_functions_frame)
        jac_entry_frame.pack(fill='x', pady=5)
        ttk.Label(jac_entry_frame, text="Jacobiano J(x,y):").grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky='w')
        ttk.Label(jac_entry_frame, text="dF1/dx =").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.text_j11_eq = tk.Text(jac_entry_frame, height=2, width=25)
        self.text_j11_eq.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(jac_entry_frame, text="dF1/dy =").grid(row=1, column=2, padx=5, pady=2, sticky='w')
        self.text_j12_eq = tk.Text(jac_entry_frame, height=2, width=25)
        self.text_j12_eq.grid(row=1, column=3, padx=5, pady=2)
        ttk.Label(jac_entry_frame, text="dF2/dx =").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        self.text_j21_eq = tk.Text(jac_entry_frame, height=2, width=25)
        self.text_j21_eq.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(jac_entry_frame, text="dF2/dy =").grid(row=2, column=2, padx=5, pady=2, sticky='w')
        self.text_j22_eq = tk.Text(jac_entry_frame, height=2, width=25)
        self.text_j22_eq.grid(row=2, column=3, padx=5, pady=2)
        self.btn_fill_example_newton_nl = ttk.Button(self.newton_custom_functions_frame, text="Usar Ejemplo en Campos Personalizados", command=self._set_custom_system_values_from_example)
        self.btn_fill_example_newton_nl.pack(pady=5)
        
        # Frame para los parámetros del método (dentro de la pestaña 1)
        params_frame = ttk.LabelFrame(config_params_frame, text="Parámetros del Método")
        params_frame.pack(pady=10, padx=10, fill='x')
        init_frame = ttk.Frame(params_frame)
        init_frame.pack(pady=5, fill='x')
        ttk.Label(init_frame, text="Vector Inicial [x₀, y₀]:").pack(side=tk.LEFT, padx=5)
        self.entry_x0_newton = ttk.Entry(init_frame, width=8); self.entry_x0_newton.pack(side=tk.LEFT, padx=5); self.entry_x0_newton.insert(0, "1.0")
        self.entry_y0_newton = ttk.Entry(init_frame, width=8); self.entry_y0_newton.pack(side=tk.LEFT, padx=5); self.entry_y0_newton.insert(0, "1.0")
        other_params_frame = ttk.Frame(params_frame)
        other_params_frame.pack(pady=5, fill='x')
        ttk.Label(other_params_frame, text="Tolerancia:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.entry_tol_newton = ttk.Entry(other_params_frame, width=10); self.entry_tol_newton.grid(row=0, column=1, padx=5, pady=2); self.entry_tol_newton.insert(0, "1e-6")
        ttk.Label(other_params_frame, text="Max. Iteraciones:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.entry_max_iter_newton = ttk.Entry(other_params_frame, width=10); self.entry_max_iter_newton.grid(row=0, column=3, padx=5, pady=2); self.entry_max_iter_newton.insert(0, "100")
        ttk.Label(other_params_frame, text="Factor Relajación (w):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.entry_w_newton = ttk.Entry(other_params_frame, width=10); self.entry_w_newton.grid(row=1, column=1, padx=5, pady=2); self.entry_w_newton.insert(0, "1.0")
        
        # Botón para resolver (dentro de la pestaña 1)
        ttk.Button(config_params_frame, text="Resolver Sistema No Lineal", 
                  command=self.solve_newton_nl_system).pack(pady=20)

        # --- Pestaña 2: Resultados Detallados ---
        results_frame = ttk.Frame(self.newton_nl_notebook, padding=10)
        self.newton_nl_notebook.add(results_frame, text='Resultados Detallados')
        
        ttk.Label(results_frame, text="Resultados de la Solución:").pack(pady=5, anchor='w')
        self.results_newton_text = tk.Text(results_frame, height=35, width=100, wrap=tk.WORD) # Ajustar tamaño según necesidad
        self.results_newton_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        self.results_newton_text.config(state=tk.DISABLED)

        # --- Pestaña 3: Resumen de Solución (NUEVA) ---
        summary_frame = ttk.Frame(self.newton_nl_notebook, padding=10)
        self.newton_nl_notebook.add(summary_frame, text='Resumen de Solución')
        
        ttk.Label(summary_frame, text="Resumen de la Solución:").pack(pady=5, anchor='w')
        self.summary_newton_text = tk.Text(summary_frame, height=15, width=100, wrap=tk.WORD, font=("Arial", 10)) # Ajustar tamaño y fuente
        self.summary_newton_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        self.summary_newton_text.config(state=tk.DISABLED)

        # --- Pestaña 4: Visualización (anteriormente Pestaña 3) ---
        plot_tab_frame = ttk.Frame(self.newton_nl_notebook, padding=10)
        self.newton_nl_notebook.add(plot_tab_frame, text='Visualización')
        
        self.newton_plot_frame = ttk.LabelFrame(plot_tab_frame, text="Gráfico del Sistema y Convergencia")
        self.newton_plot_frame.pack(pady=10, padx=10, fill='both', expand=True)
        
        # Inicializar estado de los campos de entrada personalizados
        self._toggle_newton_nl_input_mode()

    def _toggle_newton_nl_input_mode(self):
        widgets_to_toggle = [
            self.text_f1_eq, self.text_f2_eq,
            self.text_j11_eq, self.text_j12_eq,
            self.text_j21_eq, self.text_j22_eq,
            self.btn_fill_example_newton_nl
        ]

        if self.newton_nl_system_var.get() == "personalizado_2x2":
            new_state = tk.NORMAL
            # Si se cambia a personalizado y los campos están vacíos, llenarlos con el ejemplo.
            # Esto puede ser útil para el usuario.
            is_any_field_filled = any(w.get("1.0", tk.END).strip() for w in widgets_to_toggle if isinstance(w, tk.Text))
            if not is_any_field_filled:
                 self._set_custom_system_values_from_example()

        else: # "sistema_2x2" (estándar)
            new_state = tk.DISABLED
        
        for widget in widgets_to_toggle:
            if widget: # Asegurarse que el widget existe
                widget.configure(state=new_state)
    
    def _set_custom_system_values_from_example(self):
        """Establece los valores del sistema de ejemplo en los campos de texto personalizados."""
        # Sistema estándar: x**2 + xy - 10 = 0, y + 3xy**2 - 57 = 0
        # Las expresiones para eval DEBEN ser solo la parte izquierda.
        default_f1 = "x**2 + x*y - 10"
        default_f2 = "y + 3*x*y**2 - 57"
        default_j11 = "2*x + y"
        default_j12 = "x"
        default_j21 = "3*y**2"
        default_j22 = "1 + 6*x*y"

        self.text_f1_eq.delete("1.0", tk.END); self.text_f1_eq.insert("1.0", default_f1)
        self.text_f2_eq.delete("1.0", tk.END); self.text_f2_eq.insert("1.0", default_f2)
        self.text_j11_eq.delete("1.0", tk.END); self.text_j11_eq.insert("1.0", default_j11)
        self.text_j12_eq.delete("1.0", tk.END); self.text_j12_eq.insert("1.0", default_j12)
        self.text_j21_eq.delete("1.0", tk.END); self.text_j21_eq.insert("1.0", default_j21)
        self.text_j22_eq.delete("1.0", tk.END); self.text_j22_eq.insert("1.0", default_j22)

    def _set_default_system_values(self):
        """Este método está obsoleto y su funcionalidad ha sido reemplazada.
           Se deja vacío para evitar errores si es llamado accidentalmente.
        """
        pass # No hacer nada.

    def solve_newton_nl_system(self):
        # Usar una comprobación más genérica para la ventana de Newton NL
        if not self.newton_nl_window or not self.newton_nl_window.winfo_exists():
            return
            
        self._clear_text_widget(self.results_newton_text)
        self._clear_text_widget(self.summary_newton_text) # Limpiar también la nueva pestaña de resumen
        
        try:
            # Leer parámetros del método
            x0 = float(self.entry_x0_newton.get())
            y0 = float(self.entry_y0_newton.get())
            tol = float(self.entry_tol_newton.get())
            max_iter = int(self.entry_max_iter_newton.get())
            w_factor = float(self.entry_w_newton.get())
            
            # Vector inicial
            x_inicial = np.array([x0, y0])
            
            F_to_solve = None
            J_to_solve = None
            F_for_plot = None 

            if self.newton_nl_system_var.get() == "sistema_2x2":
                # print("DEBUG: Usando sistema estándar") 
                F_to_solve = newton_nl.ejemplo_F_sistema_2x2
                J_to_solve = newton_nl.ejemplo_J_sistema_2x2
                F_for_plot = newton_nl.ejemplo_F_sistema_2x2 
            elif self.newton_nl_system_var.get() == "personalizado_2x2": 
                # print("DEBUG: Usando sistema personalizado") 
                f1_str = self.text_f1_eq.get("1.0", tk.END).strip()
                f2_str = self.text_f2_eq.get("1.0", tk.END).strip()
                j11_str = self.text_j11_eq.get("1.0", tk.END).strip()
                j12_str = self.text_j12_eq.get("1.0", tk.END).strip()
                j21_str = self.text_j21_eq.get("1.0", tk.END).strip()
                j22_str = self.text_j22_eq.get("1.0", tk.END).strip()

                if not all([f1_str, f2_str, j11_str, j12_str, j21_str, j22_str]):
                    error_msg = "Error: Todas las funciones y derivadas deben ser ingresadas para el sistema personalizado."
                    self._show_error_in_text(self.results_newton_text, error_msg)
                    self._show_error_in_text(self.summary_newton_text, error_msg)
                    return

                safe_globals = {"np": np, "__builtins__": {}}
                math_funcs = {name: getattr(np, name) for name in dir(np) if isinstance(getattr(np, name), np.ufunc)}
                safe_globals.update(math_funcs)

                def F_custom(vec):
                    x, y = vec
                    local_env = {"x": x, "y": y}
                    f1 = eval(f1_str, safe_globals, local_env)
                    f2 = eval(f2_str, safe_globals, local_env)
                    return np.array([f1, f2])

                def J_custom(vec):
                    x, y = vec
                    local_env = {"x": x, "y": y}
                    j11 = eval(j11_str, safe_globals, local_env)
                    j12 = eval(j12_str, safe_globals, local_env)
                    j21 = eval(j21_str, safe_globals, local_env)
                    j22 = eval(j22_str, safe_globals, local_env)
                    return np.array([[j11, j12], [j21, j22]])
                
                F_to_solve = F_custom
                J_to_solve = J_custom
                F_for_plot = F_custom 
            else:
                error_msg = "Error: Tipo de sistema no reconocido."
                self._show_error_in_text(self.results_newton_text, error_msg)
                self._show_error_in_text(self.summary_newton_text, error_msg)
                return

            if not F_to_solve or not J_to_solve or not F_for_plot:
                error_msg = "Error: No se pudieron definir las funciones para resolver."
                self._show_error_in_text(self.results_newton_text, error_msg)
                self._show_error_in_text(self.summary_newton_text, error_msg)
                return
            
            resultado = newton_nl.resolver_sistema_newton_raphson(
                F_to_solve,
                J_to_solve,
                x_inicial,
                tol,
                max_iter,
                w_factor
            )
            
            # --- Poblar Pestaña de Resumen ---
            summary_content = ""
            if isinstance(resultado, str): # Error del backend
                summary_content = resultado
                self._show_error_in_text(self.results_newton_text, resultado) # También en detallado
            elif isinstance(resultado, dict):
                status = resultado.get('status', 'N/A')
                solucion = resultado.get('solucion')
                iter_realizadas = resultado.get('iteraciones_realizadas', 'N/A')
                norma_final = resultado.get('norma_residuo_final', float('inf'))

                summary_content += f"Estado: {status}\n"
                if solucion is not None:
                    summary_content += f"Solución final (x, y): [{solucion[0]:.10f}, {solucion[1]:.10f}]\n"
                else:
                    summary_content += "Solución final: No disponible\n"
                summary_content += f"Iteraciones realizadas: {iter_realizadas}\n"
                summary_content += f"Norma del residuo final ||F(x_k)||: {norma_final:.4e}\n"
            else:
                summary_content = "Error: Tipo de resultado inesperado del backend."
                self._show_error_in_text(self.results_newton_text, summary_content) # También en detallado
            
            self._insert_text_in_widget(self.summary_newton_text, summary_content)

            # --- Poblar Pestaña de Resultados Detallados ---
            if isinstance(resultado, dict) and 'historial_iteraciones' in resultado:
                detailed_content = "--- Historial Detallado de Iteraciones ---\n\n"
                hist = resultado['historial_iteraciones']
                
                for item in hist:
                    # Inicialización defensiva para evitar NameError si la asignación de abajo falla
                    # o si hay un flujo inesperado, o si item no es un dict.
                    norm_delta_x_val = None

                    iter_num = item['iter']
                    norm_res = item['norma_residuo']
                    x_k_val = item.get('x_k')
                    fx_k_val = item.get('Fx_k')
                    delta_k_val = item.get('delta_k')
                    jx_k_val = item.get('Jx_k')
                    
                    # Obtener norma_delta_x de forma segura
                    if isinstance(item, dict):
                        norm_delta_x_val = item.get('norma_delta_x')
                    # Si item no es un dict, norm_delta_x_val permanece None (de la inicialización)

                    detailed_content += f"--- Iteración {iter_num} ---\n"
                    
                    if x_k_val is not None:
                        detailed_content += "Vector x_k (punto de inicio de esta iteración):\n" + np.array2string(x_k_val, precision=10, suppress_small=True, separator=', ') + "\n"
                    else:
                        detailed_content += "Vector x_k: N/A\n"
                    
                    if fx_k_val is not None:
                        detailed_content += "Valor de F(x_k):\n" + np.array2string(fx_k_val, precision=10, suppress_small=True, separator=', ') + "\n"
                    else:
                        detailed_content += "Valor de F(x_k): N/A\n"
                        
                    detailed_content += f"Norma del residuo ||F(x_k)||: {norm_res:.4e}\n"

                    if jx_k_val is not None:
                        detailed_content += "Jacobiano J(x_k):\n" + np.array2string(jx_k_val, precision=5, suppress_small=True, separator=', ') + "\n" # Precisión 5 para Jacobiano
                    else:
                        detailed_content += "Jacobiano J(x_k): N/A\n"
                        
                    if delta_k_val is not None:
                        detailed_content += "Paso delta_k:\n" + np.array2string(delta_k_val, precision=10, suppress_small=True, separator=', ') + "\n"
                    else:
                        detailed_content += "Paso delta_k: N/A\n"
                    
                    if norm_delta_x_val is not None:
                        detailed_content += f"Norma del cambio ||x_{{k+1}} - x_k||: {norm_delta_x_val:.4e}\n" # x_{k+1} se calcula después de aplicar delta_k
                    else:
                        if iter_num == 0:
                             detailed_content += f"Norma del cambio ||x_{{k+1}} - x_k||: N/A (iteración inicial)\n"
                        else:
                             detailed_content += f"Norma del cambio ||x_{{k+1}} - x_k||: No calculada/Error\n"
                    
                    # Calcular y mostrar x_{k+1}
                    # Necesitamos el w_factor de la GUI para este cálculo.
                    try:
                        w_factor_gui = float(self.entry_w_newton.get()) 
                    except ValueError:
                        w_factor_gui = 1.0 # Default si la entrada es inválida, aunque el backend usa el suyo.
                                           # Esto es solo para visualización aquí.

                    if x_k_val is not None and delta_k_val is not None:
                        x_k_siguiente = x_k_val + w_factor_gui * delta_k_val
                        detailed_content += "Vector x_{k+1} (resultado de esta iteración):\n" + np.array2string(x_k_siguiente, precision=10, suppress_small=True, separator=', ') + "\n"
                    else:
                        if iter_num > 0: # No mostrar para iteración 0 si no hay delta
                            detailed_content += "Vector x_{k+1} (resultado de esta iteración): No calculable (falta x_k o delta_k)\n"

                    detailed_content += "------------------------------------\n\n"
                
                self._insert_text_in_widget(self.results_newton_text, detailed_content)
                
                if F_for_plot and resultado.get('solucion') is not None: 
                    self._plot_newton_system(F_for_plot, resultado['solucion'], hist)
            elif not isinstance(resultado, str): # Si no es un error de string y no hay historial
                 self._insert_text_in_widget(self.results_newton_text, "No hay historial detallado disponible.")
                
        except AttributeError as e_attr:
            error_msg = f"Error de Atributo: {str(e_attr)}"
            self._show_error_in_text(self.results_newton_text, error_msg)
            self._show_error_in_text(self.summary_newton_text, error_msg)
        except ValueError as e_val: # Capturar explícitamente errores de conversión de float/int
            error_msg = f"Error de valor en parámetros de entrada: {str(e_val)}\\nAsegúrese que Tolerancia, Max. Iteraciones, Relajación y Vector Inicial sean números válidos."
            self._show_error_in_text(self.results_newton_text, error_msg)
            self._show_error_in_text(self.summary_newton_text, error_msg)
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            self._show_error_in_text(self.results_newton_text, error_msg)
            self._show_error_in_text(self.summary_newton_text, error_msg)
        finally:
            if hasattr(self, 'results_newton_text') and self.results_newton_text and self.results_newton_text.winfo_exists():
                self.results_newton_text.config(state=tk.DISABLED)
            if hasattr(self, 'summary_newton_text') and self.summary_newton_text and self.summary_newton_text.winfo_exists():
                self.summary_newton_text.config(state=tk.DISABLED)

    def _plot_newton_system(self, F_func, solucion, historial):
        # print("DEBUG: Entrando a _plot_newton_system") 
        # Limpiar el frame de visualización
        for widget in self.newton_plot_frame.winfo_children():
            widget.destroy()
        
        # Crear figura y ejes
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        x_min, x_max = solucion[0] - 2, solucion[0] + 2
        y_min, y_max = solucion[1] - 2, solucion[1] + 2
        
        # Renombrar para mayor claridad y evitar conflictos de nombres si los hubiera
        x_grid_vals = np.linspace(x_min, x_max, 100)
        y_grid_vals = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid_vals, y_grid_vals)
        
        Z = np.zeros((len(y_grid_vals), len(x_grid_vals), 2))
        for i in range(len(y_grid_vals)):
            for j in range(len(x_grid_vals)):
                Z[i, j] = F_func(np.array([X[i, j], Y[i, j]]))
        
        # Graficar curvas de nivel para F1(x,y) = 0 y F2(x,y) = 0
        # Se elimina el argumento 'label' ya que no es usado por contour para la leyenda.
        ax.contour(X, Y, Z[:,:,0], levels=[0], colors='blue')
        ax.contour(X, Y, Z[:,:,1], levels=[0], colors='red')
        
        # Artistas proxy para la leyenda de las curvas de nivel
        ax.plot([], [], color='blue', lw=2, label='F1(x,y) = 0')
        ax.plot([], [], color='red', lw=2, label='F2(x,y) = 0')

        # Graficar la trayectoria de convergencia
        if len(historial) > 1: 
            x_points = [item['x_k'][0] for item in historial if item.get('x_k') is not None]
            y_points = [item['x_k'][1] for item in historial if item.get('x_k') is not None]
            if x_points and y_points: # Asegurarse que hay puntos para graficar
                ax.plot(x_points, y_points, 'g-o', markersize=4, label='Trayectoria')
        
        # Marcar la solución
        ax.plot(solucion[0], solucion[1], 'ko', markersize=8, label='Solución')
        
        # Configurar gráfica
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Sistema No Lineal y Convergencia')
        ax.legend()
        ax.grid(True)
        
        # Crear canvas para mostrar la figura
        canvas = FigureCanvasTkAgg(fig, master=self.newton_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.newton_plot_canvas = canvas # Corregido: Guardar referencia para el plot de Newton

        # # Opcional: Barra de herramientas
        # toolbar = NavigationToolbar2Tk(canvas, target_frame)
        # toolbar.update()
        # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def open_sor_window(self):
        if self.sor_window and self.sor_window.winfo_exists(): self.sor_window.lift(); return
        self.sor_window = tk.Toplevel(self.root); self.sor_window.title("Gauss-Seidel / SOR")
        self.sor_window.geometry("750x750") 
        self.sor_window.protocol("WM_DELETE_WINDOW", lambda: self._close_window_generic(self.sor_window, 'sor_window'))
        
        # Frame tamaño (arriba)
        size_frame = ttk.Frame(self.sor_window); size_frame.pack(pady=10)
        ttk.Label(size_frame, text="Tamaño (N x N):").pack(side=tk.LEFT, padx=5)
        self.sor_matrix_size_entry = ttk.Entry(size_frame, width=5); self.sor_matrix_size_entry.pack(side=tk.LEFT, padx=5); self.sor_matrix_size_entry.insert(0, "3")
        ttk.Button(size_frame, text="Generar Matriz", command=self._generate_sor_matrix_entries).pack(side=tk.LEFT, padx=5)
        
        # Nuevo Frame contenedor para A, b, x0 (centrado por defecto)
        self.sor_input_area_frame = ttk.Frame(self.sor_window)
        self.sor_input_area_frame.pack(pady=10)
        
        # Frame para parámetros (debajo del área de entrada)
        params_frame = ttk.Frame(self.sor_window); params_frame.pack(pady=10)
        ttk.Label(params_frame, text="Factor Relajación (w):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.entry_w_sor = ttk.Entry(params_frame, width=10); self.entry_w_sor.grid(row=0, column=1, padx=5, pady=2); self.entry_w_sor.insert(0, "1.0")
        ttk.Label(params_frame, text="Tolerancia (tol):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.entry_tol_sor = ttk.Entry(params_frame, width=10); self.entry_tol_sor.grid(row=1, column=1, padx=5, pady=2); self.entry_tol_sor.insert(0, "1e-6")
        ttk.Label(params_frame, text="Max. Iteraciones:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.entry_max_iter_sor = ttk.Entry(params_frame, width=10); self.entry_max_iter_sor.grid(row=2, column=1, padx=5, pady=2); self.entry_max_iter_sor.insert(0, "100")
        
        # Botón Resolver
        ttk.Button(self.sor_window, text="Resolver por Gauss-Seidel/SOR", command=self.solve_sor_system).pack(pady=10)
        
        # Área Resultados
        ttk.Label(self.sor_window, text="Resultados:").pack(pady=5)
        self.results_sor_text = tk.Text(self.sor_window, height=15, width=85); self.results_sor_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True); self.results_sor_text.config(state=tk.DISABLED)
        
        self._generate_sor_matrix_entries()
        
    def _generate_sor_matrix_entries(self):
        try: n = int(self.sor_matrix_size_entry.get()); assert 2 <= n <= 10; self.sor_matrix_size = n
        except: self._show_error_in_text(self.results_sor_text, "Tamaño de matriz inválido (2-10)."); return
        
        # Limpiar el frame contenedor principal
        if self.sor_input_area_frame:
            for widget in self.sor_input_area_frame.winfo_children(): widget.destroy()
        else: # Crear si no existe (primera vez)
             self.sor_input_area_frame = ttk.Frame(self.sor_window)
             self.sor_input_area_frame.pack(pady=10) # Asegurar que esté empaquetado

        self.entries_A_sor = []; self.entries_b_sor = []; self.entries_x0_sor = []
        
        # Crear y empaquetar frames para A, b, x0 DENTRO del frame contenedor
        matrix_a_frame = ttk.LabelFrame(self.sor_input_area_frame, text="Matriz A")
        matrix_a_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')
        
        vector_b_frame = ttk.LabelFrame(self.sor_input_area_frame, text="Vector b")
        vector_b_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')
        
        vector_x0_frame = ttk.LabelFrame(self.sor_input_area_frame, text="Estimación Inicial x0")
        vector_x0_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')

        # Poblar los frames con las entradas
        for i in range(self.sor_matrix_size):
            row_entries_a = []
            for j in range(self.sor_matrix_size):
                entry_a = ttk.Entry(matrix_a_frame, width=5)
                entry_a.grid(row=i, column=j, padx=2, pady=2)
                row_entries_a.append(entry_a)
            self.entries_A_sor.append(row_entries_a)
            
            entry_b = ttk.Entry(vector_b_frame, width=5)
            entry_b.grid(row=i, column=0, padx=2, pady=2)
            self.entries_b_sor.append(entry_b)
            
            entry_x0 = ttk.Entry(vector_x0_frame, width=8)
            entry_x0.grid(row=i, column=0, padx=2, pady=2)
            entry_x0.insert(0, "0.0")
            self.entries_x0_sor.append(entry_x0)
            
        if self.results_sor_text: self._clear_text_widget(self.results_sor_text)

    def solve_sor_system(self):
        if not self._check_window_ready(self.sor_window, self.sor_matrix_size): return
        self._clear_text_widget(self.results_sor_text)
        try:
            A_list = [[float(self.entries_A_sor[i][j].get()) for j in range(self.sor_matrix_size)] for i in range(self.sor_matrix_size)]
            b_list = [float(self.entries_b_sor[i].get()) for i in range(self.sor_matrix_size)]
            A_np = np.array(A_list, dtype=float); b_np = np.array(b_list, dtype=float)
            x0_list = []
            for i in range(self.sor_matrix_size):
                val_str = self.entries_x0_sor[i].get()
                try:
                    x0_list.append(float(val_str) if val_str else 0.0)
                except ValueError:
                    x0_list.append(0.0)
            x0_np = np.array(x0_list, dtype=float)
            w = float(self.entry_w_sor.get()); tol = float(self.entry_tol_sor.get()); max_iter = int(self.entry_max_iter_sor.get())
            resultado = nw_ray_relajacion.resolver_sistema_sor(A_np, b_np, w, tol, max_iter, x_inicial_np=x0_np)
            self._display_iterative_results(self.results_sor_text, resultado, self.sor_matrix_size, append_summary=False)
        except ValueError: self._show_error_in_text(self.results_sor_text, "Error: Valores numéricos inválidos para matrices, x0 o parámetros.")
        except Exception as e: self._show_error_in_text(self.results_sor_text, f"Error inesperado (SOR): {e}")

    def open_jacobi_window(self):
        if self.jacobi_window and self.jacobi_window.winfo_exists(): self.jacobi_window.lift(); return
        self.jacobi_window = tk.Toplevel(self.root); self.jacobi_window.title("Método de Jacobi Clásico")
        self.jacobi_window.geometry("750x750") 
        self.jacobi_window.protocol("WM_DELETE_WINDOW", lambda: self._close_window_generic(self.jacobi_window, 'jacobi_window'))
        
        # Frame tamaño (arriba)
        size_frame = ttk.Frame(self.jacobi_window); size_frame.pack(pady=10)
        ttk.Label(size_frame, text="Tamaño (N x N):").pack(side=tk.LEFT, padx=5)
        self.jacobi_matrix_size_entry = ttk.Entry(size_frame, width=5); self.jacobi_matrix_size_entry.pack(side=tk.LEFT, padx=5); self.jacobi_matrix_size_entry.insert(0, "3")
        ttk.Button(size_frame, text="Generar Matriz", command=self._generate_jacobi_matrix_entries).pack(side=tk.LEFT, padx=5)
        
        # Nuevo Frame contenedor para A, b, x0 (centrado)
        self.jacobi_input_area_frame = ttk.Frame(self.jacobi_window)
        self.jacobi_input_area_frame.pack(pady=10)
        
        # Frame para parámetros (debajo del área de entrada)
        params_frame = ttk.Frame(self.jacobi_window); params_frame.pack(pady=10)
        ttk.Label(params_frame, text="Tolerancia (tol):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.entry_tol_jacobi = ttk.Entry(params_frame, width=10); self.entry_tol_jacobi.grid(row=0, column=1, padx=5, pady=2)
        self.entry_tol_jacobi.insert(0, "1e-6")
        ttk.Label(params_frame, text="Max. Iteraciones:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.entry_max_iter_jacobi = ttk.Entry(params_frame, width=10); self.entry_max_iter_jacobi.grid(row=1, column=1, padx=5, pady=2)
        self.entry_max_iter_jacobi.insert(0, "100")
        
        # Botón Resolver
        ttk.Button(self.jacobi_window, text="Resolver por Jacobi", command=self.solve_jacobi_system).pack(pady=10)
        
        # Área Resultados
        ttk.Label(self.jacobi_window, text="Resultados:").pack(pady=5)
        self.results_jacobi_text = tk.Text(self.jacobi_window, height=15, width=85); self.results_jacobi_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True); self.results_jacobi_text.config(state=tk.DISABLED)
        
        self._generate_jacobi_matrix_entries() # Generar campos iniciales (A, b, x0)

    def _generate_jacobi_matrix_entries(self):
        try: n = int(self.jacobi_matrix_size_entry.get()); assert 2 <= n <= 10; self.jacobi_matrix_size = n
        except: self._show_error_in_text(self.results_jacobi_text, "Tamaño de matriz inválido (2-10)."); return
        
        # Limpiar el frame contenedor principal
        if self.jacobi_input_area_frame:
            for widget in self.jacobi_input_area_frame.winfo_children(): widget.destroy()
        else: # Crear si no existe
             self.jacobi_input_area_frame = ttk.Frame(self.jacobi_window)
             self.jacobi_input_area_frame.pack(pady=10)

        self.entries_A_jacobi = []; self.entries_b_jacobi = []; self.entries_x0_jacobi = []
        
        # Crear y empaquetar frames para A, b, x0 DENTRO del frame contenedor
        matrix_a_frame = ttk.LabelFrame(self.jacobi_input_area_frame, text="Matriz A")
        matrix_a_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')
        
        vector_b_frame = ttk.LabelFrame(self.jacobi_input_area_frame, text="Vector b")
        vector_b_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')
        
        vector_x0_frame = ttk.LabelFrame(self.jacobi_input_area_frame, text="Estimación Inicial x0")
        vector_x0_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')

        # Poblar los frames con las entradas
        for i in range(self.jacobi_matrix_size):
            row_entries_a = []
            for j in range(self.jacobi_matrix_size):
                entry_a = ttk.Entry(matrix_a_frame, width=5)
                entry_a.grid(row=i, column=j, padx=2, pady=2)
                row_entries_a.append(entry_a)
            self.entries_A_jacobi.append(row_entries_a)
            
            entry_b = ttk.Entry(vector_b_frame, width=5)
            entry_b.grid(row=i, column=0, padx=2, pady=2)
            self.entries_b_jacobi.append(entry_b)
            
            entry_x0 = ttk.Entry(vector_x0_frame, width=8)
            entry_x0.grid(row=i, column=0, padx=2, pady=2)
            entry_x0.insert(0, "0.0")
            self.entries_x0_jacobi.append(entry_x0)
            
        if self.results_jacobi_text: self._clear_text_widget(self.results_jacobi_text)

    def solve_jacobi_system(self):
        if not self._check_window_ready(self.jacobi_window, self.jacobi_matrix_size): return
        self._clear_text_widget(self.results_jacobi_text)
        try:
            A_list = [[float(self.entries_A_jacobi[i][j].get()) for j in range(self.jacobi_matrix_size)] for i in range(self.jacobi_matrix_size)]
            b_list = [float(self.entries_b_jacobi[i].get()) for i in range(self.jacobi_matrix_size)]
            A_np = np.array(A_list, dtype=float); b_np = np.array(b_list, dtype=float)
            x0_list = []
            for i in range(self.jacobi_matrix_size):
                val_str = self.entries_x0_jacobi[i].get()
                try:
                    x0_list.append(float(val_str) if val_str else 0.0)
                except ValueError:
                    x0_list.append(0.0)
            x0_np = np.array(x0_list, dtype=float)
            tol = float(self.entry_tol_jacobi.get()); max_iter = int(self.entry_max_iter_jacobi.get())
            dominance_message = ""
            if jacobi_method.verificar_dominancia_diagonal(A_np):
                dominance_message = "Información: La matriz ES estrictamente diagonal dominante.\nSe espera buena convergencia para Jacobi.\n---\n"
            else:
                dominance_message = "ADVERTENCIA: La matriz NO es estrictamente diagonal dominante.\nJacobi podría no converger o hacerlo lentamente.\n---\n"
            self._insert_text_in_widget(self.results_jacobi_text, dominance_message, append=False)
            resultado = jacobi_method.resolver_sistema_jacobi(A_np, b_np, tol, max_iter, x_inicial_np=x0_np)
            self._display_iterative_results(self.results_jacobi_text, resultado, self.jacobi_matrix_size, append_summary=True)
        except ValueError: self._show_error_in_text(self.results_jacobi_text, "Error: Valores numéricos inválidos para matrices, x0 o parámetros.")
        except Exception as e: self._show_error_in_text(self.results_jacobi_text, f"Error inesperado (Jacobi): {e}")

    def open_compare_linear_window(self):
        if self.compare_window and self.compare_window.winfo_exists(): self.compare_window.lift(); return
        self.compare_window = tk.Toplevel(self.root); self.compare_window.title("Comparar Métodos Lineales")
        self.compare_window.geometry("900x850") # Más alto aún para el plot
        self.compare_window.protocol("WM_DELETE_WINDOW", lambda: self._close_window_generic(self.compare_window, 'compare_window'))
        
        # Frame tamaño
        size_frame = ttk.Frame(self.compare_window); size_frame.pack(pady=10)
        ttk.Label(size_frame, text="Tamaño (N x N):").pack(side=tk.LEFT, padx=5)
        self.compare_matrix_size_entry = ttk.Entry(size_frame, width=5); self.compare_matrix_size_entry.pack(side=tk.LEFT, padx=5); self.compare_matrix_size_entry.insert(0, "3")
        ttk.Button(size_frame, text="Generar Matriz", command=self._generate_compare_matrix_entries).pack(side=tk.LEFT, padx=5)
        
        # Frame contenedor A, b, x0
        self.compare_input_area_frame = ttk.Frame(self.compare_window); self.compare_input_area_frame.pack(pady=10)
        
        # Frame parámetros (tol, max_iter, w)
        params_frame = ttk.Frame(self.compare_window); params_frame.pack(pady=5)
        ttk.Label(params_frame, text="Tolerancia (tol):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.entry_tol_comp = ttk.Entry(params_frame, width=10); self.entry_tol_comp.grid(row=0, column=1, padx=5, pady=2); self.entry_tol_comp.insert(0, "1e-6")
        ttk.Label(params_frame, text="Max. Iteraciones:").grid(row=0, column=2, padx=5, pady=2, sticky="w")
        self.entry_max_iter_comp = ttk.Entry(params_frame, width=10); self.entry_max_iter_comp.grid(row=0, column=3, padx=5, pady=2); self.entry_max_iter_comp.insert(0, "100")
        ttk.Label(params_frame, text="Factor Relajación (w para SOR):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.entry_w_comp = ttk.Entry(params_frame, width=10); self.entry_w_comp.grid(row=1, column=1, padx=5, pady=2); self.entry_w_comp.insert(0, "1.0")
        
        # Botón Ejecutar Comparación
        ttk.Button(self.compare_window, text="Ejecutar Comparación", command=self.solve_compare_linear).pack(pady=10)
        
        # --- Notebook para Resultados y Gráfico --- 
        self.compare_notebook = ttk.Notebook(self.compare_window)
        
        # Pestañas de Resultados (LU, Jacobi, SOR) - igual que antes
        frame_lu = ttk.Frame(self.compare_notebook, padding=5); self.compare_notebook.add(frame_lu, text='Factorización LU')
        self.compare_results_text_lu = tk.Text(frame_lu, height=20, width=100, wrap=tk.WORD); self.compare_results_text_lu.pack(fill=tk.BOTH, expand=True); self.compare_results_text_lu.config(state=tk.DISABLED)
        frame_jacobi = ttk.Frame(self.compare_notebook, padding=5); self.compare_notebook.add(frame_jacobi, text='Jacobi Clásico')
        self.compare_results_text_jacobi = tk.Text(frame_jacobi, height=20, width=100, wrap=tk.WORD); self.compare_results_text_jacobi.pack(fill=tk.BOTH, expand=True); self.compare_results_text_jacobi.config(state=tk.DISABLED)
        frame_sor = ttk.Frame(self.compare_notebook, padding=5); self.compare_notebook.add(frame_sor, text='Gauss-Seidel / SOR')
        self.compare_results_text_sor = tk.Text(frame_sor, height=20, width=100, wrap=tk.WORD); self.compare_results_text_sor.pack(fill=tk.BOTH, expand=True); self.compare_results_text_sor.config(state=tk.DISABLED)

        # Nueva Pestaña de Visualización
        self.compare_plot_frame = ttk.Frame(self.compare_notebook, padding=5) # Frame contenedor para el plot
        self.compare_notebook.add(self.compare_plot_frame, text='Visualización (2D/3D)')
        # El canvas de matplotlib se añadirá aquí dinámicamente

        self.compare_notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        self._generate_compare_matrix_entries() # Generar campos iniciales

    def _generate_compare_matrix_entries(self):
        try: n = int(self.compare_matrix_size_entry.get()); assert 2 <= n <= 10; self.compare_matrix_size = n
        except: self._show_error_in_text(self.compare_results_text_lu, "Tamaño inválido (2-10). Error mostrado en pestaña LU."); return
        if self.compare_input_area_frame:
            for widget in self.compare_input_area_frame.winfo_children(): widget.destroy()
        else: self.compare_input_area_frame = ttk.Frame(self.compare_window); self.compare_input_area_frame.pack(pady=10)
        self.entries_A_comp = []; self.entries_b_comp = []; self.entries_x0_comp = []
        matrix_a_frame = ttk.LabelFrame(self.compare_input_area_frame, text="Matriz A"); matrix_a_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')
        vector_b_frame = ttk.LabelFrame(self.compare_input_area_frame, text="Vector b"); vector_b_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')
        vector_x0_frame = ttk.LabelFrame(self.compare_input_area_frame, text="Estimación Inicial x0 (Jacobi/SOR)"); vector_x0_frame.pack(side=tk.LEFT, padx=10, pady=5, anchor='n')
        for i in range(self.compare_matrix_size):
            row_entries_a = []; self.entries_A_comp.append(row_entries_a)
            for j in range(self.compare_matrix_size): entry_a = ttk.Entry(matrix_a_frame, width=5); entry_a.grid(row=i, column=j, padx=2, pady=2); row_entries_a.append(entry_a)
            entry_b = ttk.Entry(vector_b_frame, width=5); entry_b.grid(row=i, column=0, padx=2, pady=2); self.entries_b_comp.append(entry_b)
            entry_x0 = ttk.Entry(vector_x0_frame, width=8); entry_x0.grid(row=i, column=0, padx=2, pady=2); entry_x0.insert(0, "0.0"); self.entries_x0_comp.append(entry_x0)
        
        # Limpiar áreas de texto y frame de plot
        self._clear_text_widget(self.compare_results_text_lu)
        self._clear_text_widget(self.compare_results_text_jacobi)
        self._clear_text_widget(self.compare_results_text_sor)
        if self.compare_plot_frame:
            for widget in self.compare_plot_frame.winfo_children(): widget.destroy()
            self.compare_plot_canvas = None # Resetear referencia al canvas
        
    def solve_compare_linear(self):
        if not self._check_window_ready(self.compare_window, self.compare_matrix_size): return
        self._clear_text_widget(self.compare_results_text_lu); self._clear_text_widget(self.compare_results_text_jacobi); self._clear_text_widget(self.compare_results_text_sor)
        # Limpiar también el frame de plot al inicio de la ejecución
        if self.compare_plot_frame:
             for widget in self.compare_plot_frame.winfo_children(): widget.destroy()
             self.compare_plot_canvas = None

        solution_x = None # Para guardar la solución y pasarla al plot
        try:
            A_list = [[float(self.entries_A_comp[i][j].get()) for j in range(self.compare_matrix_size)] for i in range(self.compare_matrix_size)]
            b_list = [float(self.entries_b_comp[i].get()) for i in range(self.compare_matrix_size)]
            x0_list = []; 
            for i in range(self.compare_matrix_size): val_str = self.entries_x0_comp[i].get(); x0_list.append(float(val_str) if val_str else 0.0)
            tol = float(self.entry_tol_comp.get()); max_iter = int(self.entry_max_iter_comp.get()); w_sor = float(self.entry_w_comp.get())
            A_np = np.array(A_list, dtype=float); b_np = np.array(b_list, dtype=float); x0_np = np.array(x0_list, dtype=float)

            # --- Ejecución LU (y guardar solución si éxito) ---
            try:
                resultado_lu = fac_LU.resolver_sistema_lu(A_np, b_np)
                if isinstance(resultado_lu, str): self._show_error_in_text(self.compare_results_text_lu, resultado_lu)
                else:
                    P, L, U, x = resultado_lu
                    solution_x = x # Guardar la solución para el plot
                    text = f"Factorización LU Exitosa:\n\nMatriz P:\n{np.array2string(P, precision=4, suppress_small=True)}\n\nL:\n{np.array2string(L, precision=4, suppress_small=True)}\n\nU:\n{np.array2string(U, precision=4, suppress_small=True)}\n\nSolución x:\n{np.array2string(x, precision=10, suppress_small=True)}"
                    self._insert_text_in_widget(self.compare_results_text_lu, text)
                    self.compare_results_text_lu.config(state=tk.DISABLED)
            except Exception as e_lu: self._show_error_in_text(self.compare_results_text_lu, f"Error ejecutando LU: {e_lu}")

            # --- Ejecución Jacobi ---
            try:
                dominance_message = "ADVERTENCIA: Matriz NO es estrictamente diagonal dominante.\n---\n" 
                if jacobi_method.verificar_dominancia_diagonal(A_np): dominance_message = "Información: Matriz ES estrictamente diagonal dominante.\n---\n"
                self._insert_text_in_widget(self.compare_results_text_jacobi, dominance_message, append=False)
                resultado_jacobi = jacobi_method.resolver_sistema_jacobi(A_np, b_np, tol, max_iter, x_inicial_np=x0_np)
                self._display_iterative_results(self.compare_results_text_jacobi, resultado_jacobi, self.compare_matrix_size, append_summary=True)
            except Exception as e_jacobi: self._show_error_in_text(self.compare_results_text_jacobi, f"Error ejecutando Jacobi: {e_jacobi}")

            # --- Ejecución SOR ---
            try:
                resultado_sor = nw_ray_relajacion.resolver_sistema_sor(A_np, b_np, w_sor, tol, max_iter, x_inicial_np=x0_np)
                self._display_iterative_results(self.compare_results_text_sor, resultado_sor, self.compare_matrix_size, append_summary=False)
            except Exception as e_sor: self._show_error_in_text(self.compare_results_text_sor, f"Error ejecutando SOR: {e_sor}")

            # --- Actualizar Pestaña de Plot --- 
            self._update_plot_tab(self.compare_matrix_size, A_np, b_np, solution_x)

        except ValueError: self._show_error_in_text(self.compare_results_text_lu, "Error: Valores numéricos inválidos...")
        except Exception as e: self._show_error_in_text(self.compare_results_text_lu, f"Error inesperado en Comparación: {e}")
        
    def _update_plot_tab(self, N, A_np, b_np, solution_x):
        # Limpiar plot anterior
        if self.compare_plot_frame:
            for widget in self.compare_plot_frame.winfo_children(): widget.destroy()
        self.compare_plot_canvas = None

        if solution_x is None:
             ttk.Label(self.compare_plot_frame, text="No se pudo obtener la solución (ej. matriz singular). No se puede graficar.").pack(padx=10, pady=10)
             return
             
        if N == 2:
            self._plot_2d_system(A_np, b_np, solution_x, self.compare_plot_frame)
        elif N == 3:
             self._plot_3d_system(A_np, b_np, solution_x, self.compare_plot_frame)
        else:
            ttk.Label(self.compare_plot_frame, text=f"Visualización gráfica no disponible para sistemas {N}x{N}.").pack(padx=10, pady=10)

    def _plot_2d_system(self, A, b, sol_x, target_frame):
        # Limpiar widgets previos en el frame
        for widget in target_frame.winfo_children(): widget.destroy()
        self.compare_plot_canvas = None # Resetear referencia

        try:
            fig, ax = plt.subplots(figsize=(6, 5)) # Crear figura y ejes

            # Rango para graficar x
            # Usar un rango alrededor de la solución x[0]
            x_range = np.linspace(sol_x[0] - 5, sol_x[0] + 5, 100) 

            # Graficar cada línea
            colors = ['r', 'g'] # Colores para las líneas
            labels = [f'Ec 1: {A[0,0]:.1f}x + {A[0,1]:.1f}y = {b[0]:.1f}', 
                      f'Ec 2: {A[1,0]:.1f}x + {A[1,1]:.1f}y = {b[1]:.1f}']
            
            for i in range(2):
                a = A[i, 0]
                b_coeff = A[i, 1] # Nombre diferente a vector b
                c = b[i]

                if np.isclose(b_coeff, 0): # Línea vertical x = c/a
                    if not np.isclose(a, 0):
                        ax.axvline(x=c/a, color=colors[i], label=labels[i], linestyle='--')
                    # Si a y b_coeff son 0, es una ecuación degenerada (0=c), no se grafica línea
                else: # Línea no vertical y = (c - a*x) / b_coeff
                    y_vals = (c - a * x_range) / b_coeff
                    ax.plot(x_range, y_vals, color=colors[i], label=labels[i])

            # Graficar el punto de solución
            ax.plot(sol_x[0], sol_x[1], 'bo', markersize=8, label=f'Solución ({sol_x[0]:.2f}, {sol_x[1]:.2f})')

            # Configuraciones del gráfico
            ax.set_title("Sistema de Ecuaciones 2x2")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True)
            ax.legend()
            # Ajustar límites para centrar la solución si es posible
            y_sol = sol_x[1]
            ax.set_xlim(sol_x[0] - 6, sol_x[0] + 6)
            ax.set_ylim(y_sol - 6, y_sol + 6) # Ajustar según la escala necesaria
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            
            fig.tight_layout() # Ajustar layout

            # Incrustar en Tkinter
            canvas = FigureCanvasTkAgg(fig, master=target_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.compare_plot_canvas = canvas # Guardar referencia

            # # Opcional: Barra de herramientas
            # toolbar = NavigationToolbar2Tk(canvas, target_frame)
            # toolbar.update()
            # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        except Exception as e:
            ttk.Label(target_frame, text=f"Error al generar gráfico 2D: {e}").pack(padx=10, pady=10)
            # Cerrar figura de matplotlib si se creó pero falló después
            plt.close(fig) 

    def _plot_3d_system(self, A, b, sol_x, target_frame):
        # Limpiar frame
        for widget in target_frame.winfo_children(): widget.destroy()
        self.compare_plot_canvas = None
        
        try:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Definir rango para x, y alrededor de la solución
            # El rango puede necesitar ajustes dependiendo del sistema
            margin = 3 
            x_surf = np.linspace(sol_x[0] - margin, sol_x[0] + margin, 20)
            y_surf = np.linspace(sol_x[1] - margin, sol_x[1] + margin, 20)
            x_surf, y_surf = np.meshgrid(x_surf, y_surf)

            # Colores y etiquetas
            colors = ['r', 'g', 'c'] # Rojo, Verde, Cyan
            alphas = 0.5 # Transparencia para los planos

            # Graficar cada plano
            for i in range(3):
                a, b_coeff, c_coeff = A[i, 0], A[i, 1], A[i, 2]
                d = b[i]

                if not np.isclose(c_coeff, 0): # Plano normal: z = (d - ax - by) / c
                    z_surf = (d - a*x_surf - b_coeff*y_surf) / c_coeff
                    ax.plot_surface(x_surf, y_surf, z_surf, color=colors[i], alpha=alphas, 
                                    label=f'Ec {i+1}', rstride=100, cstride=100) # rstride/cstride bajos para visualización
                elif not np.isclose(b_coeff, 0): # Plano vertical en y: y = (d - ax - cz) / b
                    # Graficar planos verticales es más complejo. 
                    # Podríamos omitirlo o usar otra técnica, aquí lo omitimos por simplicidad.
                    print(f"Omitiendo plano {i+1} (vertical en y, z variable) para simplificar.")
                    # Alternativa: plot_trisurf si tuviéramos puntos
                elif not np.isclose(a, 0): # Plano vertical en x: x = (d - by - cz) / a
                     print(f"Omitiendo plano {i+1} (vertical en x, z variable) para simplificar.")
                # Si a,b,c son 0, es degenerado.

            # Graficar el punto de solución
            ax.scatter(sol_x[0], sol_x[1], sol_x[2], color='b', marker='o', s=100, label='Solución', depthshade=True)

            # Configuraciones
            ax.set_title("Sistema de Ecuaciones 3x3 (Planos)")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # Añadir leyenda (puede ser difícil de ver con plot_surface)
            # ax.legend() 

            # Ajustar límites (puede ser necesario ajustar el 'margin' arriba)
            ax.set_xlim(sol_x[0] - margin, sol_x[0] + margin)
            ax.set_ylim(sol_x[1] - margin, sol_x[1] + margin)
            ax.set_zlim(sol_x[2] - margin, sol_x[2] + margin)
            
            fig.tight_layout()

            # Incrustar
            canvas = FigureCanvasTkAgg(fig, master=target_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.compare_plot_canvas = canvas

        except Exception as e:
            ttk.Label(target_frame, text=f"Error al generar gráfico 3D: {e}").pack(padx=10, pady=10)
            plt.close(fig)

    # --- Funciones auxiliares GUI (existentes) ---
    def _show_error_in_text(self, text_widget, message):
        if text_widget:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, message + "\n")
            text_widget.config(state=tk.DISABLED)

    def _clear_text_widget(self, text_widget):
        if text_widget:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            # No lo deshabilitamos aquí, lo hará la función que inserta el resultado o error

    def _insert_text_in_widget(self, text_widget, message, append=False):
        """Inserta texto en el widget. Si append es False, limpia antes."""
        if not text_widget: return
        text_widget.config(state=tk.NORMAL)
        if not append:
            text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, message)
        # No se deshabilita aquí, se hará al final de la operación principal

    def _check_window_ready(self, window, matrix_size):
        if not window or not window.winfo_exists() or matrix_size == 0:
            return False
        return True

    def _display_iterative_results(self, text_widget, resultado, matrix_size_for_log_line_width, append_summary=False):
        if not text_widget: return
        # Si no estamos añadiendo a texto existente, limpiamos.
        # Si append_summary es True, asumimos que ya hay un mensaje (ej. dominancia) y queremos añadir a eso.
        if not append_summary: 
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
        
        current_text_content = text_widget.get(1.0, tk.END) if append_summary else ""
        text_widget.config(state=tk.NORMAL)
        if not append_summary: # Solo borrar si no estamos añadiendo explícitamente
             text_widget.delete(1.0, tk.END)
        else: # Si estamos añadiendo, asegurar que el cursor esté al final para el nuevo texto.
            text_widget.mark_set("insert", tk.END)

        full_message = ""
        if isinstance(resultado, str): # Error de validación inicial del backend
            full_message = resultado + "\n"
        elif isinstance(resultado, dict):
            summary = (
                f"Estado: {resultado.get('status', 'N/A')}\n\n"
                f"Solución final x:\n{np.array2string(resultado.get('solucion', np.array([])), precision=10, suppress_small=True)}\n\n"
                f"Iteraciones realizadas: {resultado.get('iteraciones_realizadas', 'N/A')}\n"
                f"Norma final del residuo: {resultado.get('norma_residuo_final', float('inf')):.2e}\n\n"
                "--- Historial de Iteraciones ---\n"
            )
            full_message += summary
            historial = resultado.get('historial_iteraciones', [])
            if not historial:
                full_message += "(No hay historial detallado disponible)\n"
            else:
                header = f"{'Iter':<5} | {'Norma Residuo':<15} | Vector x_k\n"
                full_message += header
                line_len_approx = 5 + 15 + 3 + matrix_size_for_log_line_width * 10 
                full_message += "-" * min(line_len_approx, 80) + "\n"
                for item in historial:
                    iter_num = item.get('iter', '-')
                    norm_res = item.get('norma_residuo', float('nan'))
                    x_k_vec = item.get('x_k', np.array([]))
                    x_k_str_oneline = ' '.join(np.array2string(x_k_vec, precision=10, suppress_small=True, separator=', ').replace('\n', '').split())
                    line = f"{iter_num:<5} | {norm_res:<15.4e} | {x_k_str_oneline}\n"
                    full_message += line
        else:
            full_message = "Error: Tipo de resultado inesperado del resolvedor.\n"
        
        if append_summary:
            text_widget.insert(tk.END, full_message) # Añade al final del texto existente
        else:
            text_widget.insert(1.0, full_message) # Reemplaza todo el texto

        text_widget.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop() 