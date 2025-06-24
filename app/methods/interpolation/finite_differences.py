"""
Método de Diferencias Finitas para Aproximación de Derivadas

Este script implementa el método numérico conocido como "Diferencias Finitas",
utilizado para aproximar la derivada de una función a partir de un conjunto
de puntos discretos igualmente espaciados.

Tipos de aproximaciones usadas:

1. Diferencia Progresiva (Sucesiva):
   Aproxima la derivada usando el punto actual y el siguiente.
   Fórmula: f'(x_i) ≈ (f(x_{i+1}) - f(x_i)) / h

2. Diferencia Regresiva:
   Usa el punto actual y el anterior.
   Fórmula: f'(x_i) ≈ (f(x_i) - f(x_{i-1})) / h

3. Diferencia Central (Centrada):
   Usa el punto anterior y el siguiente.
   Fórmula: f'(x_i) ≈ (f(x_{i+1}) - f(x_{i-1})) / (2h)

Este método es ampliamente utilizado en análisis numérico, simulaciones, y
problemas computacionales cuando se dispone únicamente de datos discretos
y no de una función analítica.

Limitación:
- Los valores de x deben estar igualmente espaciados (h constante).
"""

import numpy as np

def is_spacing_equal(x_vals, tolerance=1e-9):
    """Verifica si los valores en el array x_vals están igualmente espaciados."""
    if len(x_vals) < 2:
        return True # Consideramos un solo punto o ninguno como espaciado igual
    diffs = np.diff(x_vals)
    return np.all(np.isclose(diffs, diffs[0], atol=tolerance))

def numerical_derivatives_with_formulas(x_vals, y_vals):
    """
    Calcula las derivadas numéricas (progresiva, regresiva, central) para un conjunto
    de puntos (x_vals, y_vals) igualmente espaciados.
    Retorna una lista de diccionarios, cada uno con las derivadas para un punto x.
    """
    if not is_spacing_equal(x_vals):
        raise ValueError("Los valores de x no están igualmente espaciados.")
    
    if len(x_vals) != len(y_vals):
        raise ValueError("Los arrays x_vals e y_vals deben tener la misma longitud.")

    if len(x_vals) < 2: # Se necesitan al menos dos puntos para la mayoría de las diferencias
        # Podríamos retornar vacío o manejarlo de otra forma.
        # Por ahora, si es menos de 2 puntos, no se pueden calcular muchas diferencias.
        # Para un solo punto, ninguna diferencia es calculable.
        # Para el caso de la GUI, ya se valida > 2 puntos.
        return []


    h = x_vals[1] - x_vals[0]
    n = len(x_vals)
    results = []

    for i in range(n):
        punto_x = x_vals[i]
        result = {'x': punto_x, 'sucesiva': None, 'central': None, 'regresiva': None}

        # Sucesiva: Necesita x[i] y x[i+1]
        if i < n - 1:
            f_x_s = y_vals[i]
            f_xh_s = y_vals[i+1]
            formula_s = f"(f({x_vals[i+1]}) - f({x_vals[i]})) / {h:.4g} = ({f_xh_s:.4g} - {f_x_s:.4g}) / {h:.4g}"
            resultado_s = (f_xh_s - f_x_s) / h
            result['sucesiva'] = (formula_s, resultado_s)

        # Regresiva: Necesita x[i] y x[i-1]
        if i > 0:
            f_x_r = y_vals[i]
            f_xp_r = y_vals[i-1]
            formula_r = f"(f({x_vals[i]}) - f({x_vals[i-1]})) / {h:.4g} = ({f_x_r:.4g} - {f_xp_r:.4g}) / {h:.4g}"
            resultado_r = (f_x_r - f_xp_r) / h
            result['regresiva'] = (formula_r, resultado_r)
            
        # Central: Necesita x[i-1] y x[i+1]
        if i > 0 and i < n - 1:
            f_xp_c = y_vals[i-1]
            f_xh_c = y_vals[i+1]
            formula_c = f"(f({x_vals[i+1]}) - f({x_vals[i-1]})) / (2*{h:.4g}) = ({f_xh_c:.4g} - {f_xp_c:.4g}) / {2*h:.4g}"
            resultado_c = (f_xh_c - f_xp_c) / (2*h)
            result['central'] = (formula_c, resultado_c)
            
        results.append(result)

    return results

if __name__ == '__main__':
    print(f"--- Ejecutando pruebas para: finite_differences.py ---")
    
    # Datos de entrada para prueba
    x_test_fd = np.array([1, 2, 3, 4, 5], dtype=float)
    y_test_fd = np.array([1, 4, 9, 16, 25], dtype=float)  # f(x) = x^2

    print(f"Probando con x = {x_test_fd}, y = {y_test_fd}")
    
    if is_spacing_equal(x_test_fd):
        resultados_test_fd = numerical_derivatives_with_formulas(x_test_fd, y_test_fd)
        if resultados_test_fd:
            for r_test in resultados_test_fd:
                print(f"Punto x = {r_test['x']:.1f}")
                if r_test.get('sucesiva'):
                    print(f"  Sucesiva:  Resultado = {r_test['sucesiva'][1]:.4f} (Fórmula: {r_test['sucesiva'][0]})")
                if r_test.get('central'):
                    print(f"  Central:   Resultado = {r_test['central'][1]:.4f} (Fórmula: {r_test['central'][0]})")
                if r_test.get('regresiva'):
                    print(f"  Regresiva: Resultado = {r_test['regresiva'][1]:.4f} (Fórmula: {r_test['regresiva'][0]})")
                print("-" * 20)
        else:
            print("No se generaron resultados (verifique la cantidad de puntos).")
    else:
        print("Error en datos de prueba: Los valores de x no están igualmente espaciados.")
    
    print(f"--- Fin de pruebas para: finite_differences.py ---")
