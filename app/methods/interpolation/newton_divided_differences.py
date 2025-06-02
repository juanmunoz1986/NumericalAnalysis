"""
Método de Interpolación de Newton por Diferencias Divididas

Descripción:
Este método permite construir un polinomio que interpola un conjunto de puntos dados (x, y)
sin necesidad de que estén igualmente espaciados. Utiliza una tabla en forma de pirámide donde
se calculan las diferencias divididas. El polinomio se construye en forma progresiva a partir
de estos coeficientes.

Fórmula general del polinomio de Newton:
P(x) = a0 + a1(x - x0) + a2(x - x0)(x - x1) + ... + an(x - x0)(x - x1)...(x - xn-1)

Aplicaciones:
- Interpolación numérica
- Aproximación de funciones a partir de datos discretos
- Modelado de datos experimentales

Requisitos:
- No requiere que los puntos estén igualmente espaciados
"""

import numpy as np

def build_divided_difference_table(x_vals, y_vals):
    """
    Construye la tabla de diferencias divididas de Newton.
    Retorna una tabla con los valores numéricos.
    """
    n = len(x_vals)
    if n != len(y_vals):
        raise ValueError("Los arrays x_vals e y_vals deben tener la misma longitud.")
    if n == 0:
        return []
        
    # Inicializar la tabla con ceros o None, y luego llenar la primera columna
    # Usaremos una lista de listas para la tabla, y y_vals para la primera columna.
    # La diagonal superior contendrá los coeficientes b0, b1, ..., bn
    # La tabla comúnmente se visualiza como triangular.
    # Para el cálculo, podemos usar una copia de y_vals e ir actualizando.

    coef_table = np.zeros([n, n])
    coef_table[:,0] = np.copy(y_vals) # Primera columna son los y_i

    for j in range(1, n): # Columna de la diferencia dividida (orden j)
        for i in range(n - j): # Fila (comienza en 0 hasta n-j-1)
            # f[xi, ..., x_{i+j}] = (f[x_{i+1}, ..., x_{i+j}] - f[xi, ..., x_{i+j-1}]) / (x_{i+j} - xi)
            numerator = coef_table[i+1, j-1] - coef_table[i, j-1]
            denominator = x_vals[i+j] - x_vals[i]
            if np.isclose(denominator, 0.0):
                raise ValueError(f"Error: División por cero detectada. x_vals[{i+j}] y x_vals[{i}] son muy cercanos o iguales.")
            coef_table[i,j] = numerator / denominator
            
    # La tabla completa (coef_table) se puede retornar si se desea visualizarla entera.
    # Para este ejemplo, vamos a retornar la tabla como una lista de listas
    # tal como la GUI podría necesitarla (o la parte relevante).
    # La función original retornaba la tabla completa y también las fórmulas.
    # Por simplicidad, esta versión del backend solo retornará la tabla de valores.
    # La GUI es la que tiene la lógica más compleja para mostrar tablas ahora.
    # Para que coincida con lo que la GUI espera (value_table, formula_table), 
    # podemos retornar la tabla de coeficientes y un placeholder para las fórmulas,
    # o adaptar la GUI.
    # Para esta corrección, nos enfocamos en que el backend funcione.
    # La función print_divided_difference_table del script original no se usa en la GUI.

    # Convertir la tabla de coeficientes (NumPy array) a una lista de listas de Python
    # La GUI espera que la tabla tenga strings '' para celdas no usadas.
    # La tabla `coef_table` tiene los coeficientes en la parte superior izquierda.
    # value_table[i][j] para la GUI.
    value_table_gui_format = [["" for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n - r):
            value_table_gui_format[r][c] = coef_table[r,c]
            
    # La función original también retornaba una tabla de fórmulas.
    # Por ahora, para simplificar la corrección del error de print, no la generaremos aquí.
    # Si la GUI la necesita, se deberá añadir esa lógica o modificar la GUI.
    formula_table_placeholder = [["" for _ in range(n)] for _ in range(n)] 

    return value_table_gui_format, formula_table_placeholder # Devolver en el formato que espera la GUI

def get_newton_coefficients(value_table):
    """
    Extrae los coeficientes del polinomio (la diagonal superior) desde la tabla de diferencias divididas.
    value_table[0][0] es a0, value_table[0][1] es a1, etc.
    """
    if not value_table:
        return []
    n = len(value_table[0]) # Asumiendo tabla cuadrada o al menos con la primera fila completa
    # Los coeficientes son la primera fila de la tabla generada por build_divided_difference_table
    # (después de ser formateada como value_table_gui_format)
    coeffs = []
    for j in range(n):
        if value_table[0][j] != "": # Tomar hasta donde haya valores
            coeffs.append(value_table[0][j])
        else:
            break # Detenerse si encontramos una celda vacía en la primera fila
    return coeffs

def evaluate_newton_polynomial(x_data, coefficients, x_eval):
    """
    Evalúa el polinomio de Newton en un punto dado x_eval.
    x_data: Los puntos x originales usados para generar los coeficientes.
    coefficients: Los coeficientes a0, a1, ..., an.
    x_eval: El punto donde evaluar el polinomio.
    """
    n = len(coefficients)
    if n == 0:
        return 0 # O None, o lanzar error
    
    # P(x) = a0 + a1(x - x0) + a2(x - x0)(x - x1) + ...
    # Se puede evaluar eficientemente usando la forma anidada (Algoritmo de Horner generalizado)
    # P(x) = (...(an(x - x_{n-1}) + a_{n-1})(x-x_{n-2}) + ... + a1)(x-x0) + a0
    # Aquí los x_data son los x_i originales.
    
    result = coefficients[n-1] # Empezar con an
    for i in range(n - 2, -1, -1):
        if i >= len(x_data):
             raise ValueError(f"Índice x_data[{i}] fuera de rango. Se necesitan al menos {n-1} puntos x_data para {n} coeficientes.")
        result = result * (x_eval - x_data[i]) + coefficients[i]
    return result

if __name__ == '__main__':
    print(f"--- Ejecutando pruebas para: newton_divided_differences.py ---")
    # Datos de entrada para prueba
    x_test_ndd = np.array([0, 0.5, 1, 1.5, 2], dtype=float)
    y_test_ndd = np.array([0, 2.5, 5.8, 8.5, 10.2], dtype=float)

    print(f"Probando con x = {x_test_ndd}, y = {y_test_ndd}")

    # Construir tabla de diferencias divididas
    # La función ahora devuelve dos tablas, la segunda es un placeholder para las fórmulas.
    val_table_test, _ = build_divided_difference_table(x_test_ndd, y_test_ndd)

    print("\nTabla de Diferencias Divididas (Valores) - Prueba:")
    # Para imprimir la tabla de forma similar a la original, necesitamos una función print.
    # Reutilizaremos una versión simplificada aquí para la prueba.
    def print_simple_table(table_data, x_coords):
        if not table_data: print("(Tabla vacía)"); return
        num_cols = len(table_data[0])
        headers = ["x_i"] + [f"D{j}" for j in range(num_cols)]
        print(" | ".join(f"{h:^12}" for h in headers[:len(x_coords)+1])) # Ajustar headers a datos
        print("-" * (15 * (len(x_coords)+1)))
        for i in range(len(x_coords)):
            row_str = f"{x_coords[i]:^12.4g} | "
            for j in range(num_cols - i):
                val = table_data[i][j]
                if isinstance(val, (float, np.float_)) and val != "":
                    row_str += f"{val:^12.4f} | "
                else:
                    row_str += f"{"":^12} | " 
            print(row_str.strip().strip('|').strip())
        print("")

    if val_table_test:
        # La tabla devuelta por build_divided_difference_table es `value_table_gui_format`
        # Necesitamos transponerla o reinterpretarla para la impresión estilo pirámide original
        # o simplemente imprimir la diagonal que son los coeficientes.
        # La función build_divided_difference_table ahora devuelve la tabla tal como la GUI la espera.
        # La impresión aquí será sobre esa tabla.
        print_simple_table(val_table_test, x_test_ndd)
    else:
        print("(No se generó la tabla de valores)")

    # Obtener los coeficientes del polinomio
    coefficients_test_ndd = get_newton_coefficients(val_table_test)
    print("Coeficientes del Polinomio de Newton (Prueba):")
    print(np.array(coefficients_test_ndd))

    # Evaluar el polinomio para un valor dado de x
    x_eval_point = 5.0
    y_interpolated_test = evaluate_newton_polynomial(x_test_ndd, coefficients_test_ndd, x_eval_point)
    print(f"\nInterpolación en x = {x_eval_point} (Prueba): P(x) = {y_interpolated_test:.4f}")
    
    # Prueba con x_eval_point = 1.25 (como en la GUI)
    x_eval_point_gui = 1.25
    y_interpolated_test_gui = evaluate_newton_polynomial(x_test_ndd, coefficients_test_ndd, x_eval_point_gui)
    print(f"Interpolación en x = {x_eval_point_gui} (Prueba GUI): P(x) = {y_interpolated_test_gui:.4f}")
    
    print(f"--- Fin de pruebas para: newton_divided_differences.py ---")
