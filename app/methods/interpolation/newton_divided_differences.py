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

def build_divided_difference_table(x, y):
    """
    Construye la tabla de diferencias divididas de Newton.
    Retorna dos tablas: una con los valores numéricos y otra con las fórmulas utilizadas.
    """
    n = len(x)
    value_table = [["" for _ in range(n)] for _ in range(n)]
    formula_table = [["" for _ in range(n)] for _ in range(n)]

    # Primera columna: valores originales de y
    for i in range(n):
        value_table[i][0] = y[i]
        formula_table[i][0] = f"{y[i]:.4f}"

    # Calcular las diferencias divididas
    for j in range(1, n):
        for i in range(n - j):
            a = value_table[i + 1][j - 1]
            b = value_table[i][j - 1]
            xa = x[i + j]
            xb = x[i]

            numerator = f"[({a:.2f}) - ({b:.2f})]"
            denominator = f"[({xa}) - ({xb})]"
            formula = f"{numerator} / {denominator}"

            value = (a - b) / (xa - xb)
            value_table[i][j] = value
            formula_table[i][j] = formula

    return value_table, formula_table

def print_divided_difference_table(table, title=""):
    """
    Imprime una tabla de diferencias divididas con formato amigable.
    """
    n = len(table)
    headers = ["x", "y (a0)"] + [f"a{j}" for j in range(1, n)]
    print(title)
    print(" | ".join(f"{h:^40}" for h in headers))
    print("-" * 44 * len(headers))

    for i in range(n):
        row = [f"{x[i]:^40}"]  # Primera columna: valores de x
        for j in range(n):
            val = table[i][j]
            text = f"{val:.4f}" if isinstance(val, float) else str(val)
            row.append(f"{text:^40}")
        print(" | ".join(row))
    print("\n")

def get_newton_coefficients(table):
    """
    Extrae los coeficientes del polinomio desde la primera fila de la tabla.
    """
    return [table[0][j] for j in range(len(table[0])) if table[0][j] != ""]

def evaluate_newton_polynomial(x, coef, x_eval):
    """
    Evalúa el polinomio de Newton en un punto dado x_eval.
    """
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_eval - x[i]) + coef[i]
    return result

# Datos de entrada (puedes modificar estos valores)
x = np.array([0, 0.5, 1, 1.5, 2])
y = np.array([0, 2.5, 5.8, 8.5, 10.2])

# Construir tabla de diferencias divididas
value_table, formula_table = build_divided_difference_table(x, y)

print("Divided Difference Table - VALUES:")
print_divided_difference_table(value_table)

print("Divided Difference Table - FORMULAS:")
print_divided_difference_table(formula_table)

# Obtener los coeficientes del polinomio
coefficients = get_newton_coefficients(value_table)
print("\nNewton Polynomial Coefficients:")
print(np.array(coefficients))

# Evaluar el polinomio para un valor dado de x
x_value = 5
y_value = evaluate_newton_polynomial(x, coefficients, x_value)
print(f"\nInterpolación en x = {x_value}: y = {y_value:.4f}")
