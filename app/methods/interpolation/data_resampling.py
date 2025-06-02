import numpy as np
from . import newton_divided_differences # Usamos importación relativa

def resample_data_newton(x_original: np.ndarray, 
                         y_original: np.ndarray, 
                         x_target_start: float, 
                         x_target_end: float, 
                         new_h: float):
    """
    Re-muestrea los datos (x_original, y_original) para un nuevo conjunto de puntos x
    definidos por un rango [x_target_start, x_target_end] y un nuevo paso new_h,
    utilizando la interpolación polinómica de Newton.

    Args:
        x_original (np.ndarray): Valores x originales.
        y_original (np.ndarray): Valores y originales correspondientes a x_original.
        x_target_start (float): Valor inicial para los nuevos puntos x re-muestreados.
        x_target_end (float): Valor final para los nuevos puntos x re-muestreados.
        new_h (float): Nuevo paso deseado entre los puntos x re-muestreados.

    Returns:
        tuple[np.ndarray, np.ndarray]: Una tupla conteniendo:
            - new_x_points (np.ndarray): Los nuevos puntos x generados.
            - new_y_points (np.ndarray): Los valores y interpolados en new_x_points.
            
    Raises:
        ValueError: Si new_h es cero o negativo, o si x_target_start > x_target_end.
        ValueError: Si los arrays originales x e y no tienen la misma longitud o están vacíos.
        ValueError: Si no se pueden construir los coeficientes de Newton (e.g., x_original no es adecuado).
    """
    if not isinstance(x_original, np.ndarray):
        x_original = np.array(x_original, dtype=float)
    if not isinstance(y_original, np.ndarray):
        y_original = np.array(y_original, dtype=float)

    if len(x_original) != len(y_original):
        raise ValueError("Los arrays x_original e y_original deben tener la misma longitud.")
    if len(x_original) == 0:
        raise ValueError("Los arrays x_original e y_original no pueden estar vacíos.")
        
    if new_h <= 0:
        raise ValueError("El nuevo paso (new_h) debe ser positivo.")
    if x_target_start > x_target_end:
        raise ValueError("x_target_start no puede ser mayor que x_target_end.")

    # 1. Construir el polinomio de Newton con (x_original, y_original)
    try:
        value_table, _ = newton_divided_differences.build_divided_difference_table(x_original, y_original)
        coefficients = newton_divided_differences.get_newton_coefficients(value_table)
    except Exception as e:
        raise ValueError(f"Error al construir el polinomio de Newton: {e}")

    if not coefficients: # o len(coefficients) == 0
         raise ValueError("No se pudieron obtener los coeficientes del polinomio de Newton.")

    # 2. Generar los new_x_points
    # Calculamos el número de puntos para np.linspace para incluir el punto final correctamente.
    if x_target_start == x_target_end: # Caso de un solo punto
        num_points = 1
    else:
        num_points = int(round((x_target_end - x_target_start) / new_h)) + 1
        if num_points <= 0: # Si h es muy grande comparado con el rango
            num_points = 1 # Al menos un punto

    new_x_points = np.linspace(x_target_start, x_target_end, num_points)
    
    # Si num_points fue 1, linspace podría haber creado un array con x_target_start
    # Asegurémonos de que si es un solo punto, sea x_target_start
    if num_points == 1:
        new_x_points = np.array([x_target_start])


    # 3. Evaluar el polinomio en new_x_points para obtener new_y_points
    new_y_points = np.array([newton_divided_differences.evaluate_newton_polynomial(x_original, coefficients, x_val) for x_val in new_x_points])

    return new_x_points, new_y_points

if __name__ == '__main__':
    # Ejemplo de uso y prueba
    print("--- Probando data_resampling.py ---")
    
    # Datos originales (ej: f(x) = x^2)
    x_orig_test = np.array([0, 1, 2, 3, 4], dtype=float)
    y_orig_test = np.array([0, 1, 4, 9, 16], dtype=float)
    
    print(f"Original x: {x_orig_test}")
    print(f"Original y: {y_orig_test}")

    # Parámetros para re-muestreo
    x_start_test = 0.5
    x_end_test = 3.5
    h_new_test = 0.25
    
    print(f"Remuestreando de x={x_start_test} a x={x_end_test} con h={h_new_test}")

    try:
        x_resampled, y_resampled = resample_data_newton(x_orig_test, y_orig_test, x_start_test, x_end_test, h_new_test)
        print(f"Nuevos x: {x_resampled}")
        print(f"Nuevos y: {y_resampled}")
        
        # Verificación de los nuevos x
        if len(x_resampled) > 1:
            print(f"  Primer x nuevo: {x_resampled[0]:.4f}, Último x nuevo: {x_resampled[-1]:.4f}, Longitud: {len(x_resampled)}")
            print(f"  Paso h real en nuevos x (aprox): {(x_resampled[1]-x_resampled[0]):.4f}")

        # Prueba con un rango que solo da un punto
        print("\\nProbando re-muestreo a un solo punto:")
        x_single, y_single = resample_data_newton(x_orig_test, y_orig_test, 1.5, 1.5, 0.1)
        print(f"x_single: {x_single}, y_single: {y_single}") # Debería ser P(1.5)

        print("\\nProbando con h muy grande:")
        x_h_large, y_h_large = resample_data_newton(x_orig_test, y_orig_test, 0, 4, 5.0) # h > rango
        print(f"x_h_large: {x_h_large}, y_h_large: {y_h_large}") # Debería dar al menos el punto inicial.

    except ValueError as ve:
        print(f"Error durante la prueba: {ve}")
    except Exception as e:
        print(f"Una excepción inesperada ocurrió: {e}")
        
    print("--- Fin de pruebas data_resampling.py ---") 