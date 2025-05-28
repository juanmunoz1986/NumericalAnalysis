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

# Datos de entrada
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([1, 4, 9, 16, 25], dtype=float)  # f(x) = x^2

# Verificar espaciado uniforme
def is_spacing_equal(x):
    diffs = np.diff(x)
    return np.all(diffs == diffs[0])

# Derivadas con fórmula mostrada
def numerical_derivatives_with_formulas(x, y):
    if not is_spacing_equal(x):
        raise ValueError("Los valores de x no están igualmente espaciados.")

    h = x[1] - x[0]
    n = len(x)
    results = []

    for i in range(n):
        punto_x = x[i]
        punto_y = y[i]

        result = {'x': punto_x}

        if i == 0:
            # Solo sucesiva
            f_x = y[i]
            f_xh = y[i+1]
            formula = f"(f({x[i+1]}) - f({x[i]})) / h = ({f_xh} - {f_x}) / {h}"
            resultado = (f_xh - f_x) / h
            result['sucesiva'] = (formula, resultado)
            result['central'] = None
            result['regresiva'] = None

        elif i == n - 1:
            # Solo regresiva
            f_x = y[i]
            f_xh = y[i-1]
            formula = f"(f({x[i]}) - f({x[i-1]})) / h = ({f_x} - {f_xh}) / {h}"
            resultado = (f_x - f_xh) / h
            result['sucesiva'] = None
            result['central'] = None
            result['regresiva'] = (formula, resultado)

        else:
            # Sucesiva
            f_x = y[i]
            f_xh = y[i+1]
            formula_s = f"(f({x[i+1]}) - f({x[i]})) / h = ({f_xh} - {f_x}) / {h}"
            resultado_s = (f_xh - f_x) / h

            # Regresiva
            f_xp = y[i-1]
            formula_r = f"(f({x[i]}) - f({x[i-1]})) / h = ({f_x} - {f_xp}) / {h}"
            resultado_r = (f_x - f_xp) / h

            # Central
            formula_c = f"(f({x[i+1]}) - f({x[i-1]})) / (2h) = ({f_xh} - {f_xp}) / {2*h}"
            resultado_c = (f_xh - f_xp) / (2*h)

            result['sucesiva'] = (formula_s, resultado_s)
            result['central'] = (formula_c, resultado_c)
            result['regresiva'] = (formula_r, resultado_r)

        results.append(result)

    return results

# Mostrar resultados
if is_spacing_equal(x):
    resultados = numerical_derivatives_with_formulas(x, y)
    for r in resultados:
        print(f"x = {r['x']:.1f}")
        if r['sucesiva']:
            print("  Sucesiva:")
            print(f"    Fórmula:  {r['sucesiva'][0]}")
            print(f"    Resultado: {r['sucesiva'][1]}")
        if r['central']:
            print("  Central:")
            print(f"    Fórmula:  {r['central'][0]}")
            print(f"    Resultado: {r['central'][1]}")
        if r['regresiva']:
            print("  Regresiva:")
            print(f"    Fórmula:  {r['regresiva'][0]}")
            print(f"    Resultado: {r['regresiva'][1]}")
        print()
else:
    print("Los valores de x no están igualmente espaciados.")
