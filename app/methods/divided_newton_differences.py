import numpy as np

def construir_piramide(x, y):
    n = len(x)
    tabla_valores = [["" for _ in range(n)] for _ in range(n)]
    tabla_formulas = [["" for _ in range(n)] for _ in range(n)]

    # Primera columna: valores de y
    for i in range(n):
        tabla_valores[i][0] = y[i]
        tabla_formulas[i][0] = f"{y[i]:.4f}"

    # Calcular diferencias divididas
    for j in range(1, n):
        for i in range(n - j):
            a = tabla_valores[i + 1][j - 1]
            b = tabla_valores[i][j - 1]
            xa = x[i + j]
            xb = x[i]

            numerador = f"[({a:.2f}) - ({b:.2f})]"
            denominador = f"[({xa}) - ({xb})]"
            formula = f"{numerador} / {denominador}"

            valor = (a - b) / (xa - xb)
            tabla_valores[i][j] = valor
            tabla_formulas[i][j] = formula

    return tabla_valores, tabla_formulas

def imprimir_piramide(tabla, encabezado=""):
    n = len(tabla)
    # Crear encabezado dinámico
    headers = ["x", "y (a0)"] + [f"a{j}" for j in range(1, n)]
    print(encabezado)
    print(" | ".join(f"{h:^40}" for h in headers))
    print("-" * 44 * len(headers))

    for i in range(n):
        fila = [f"{x[i]:^40}"]  # Primera columna: x[i]
        for j in range(n):
            val = tabla[i][j]
            texto = f"{val:.4f}" if isinstance(val, float) else str(val)
            fila.append(f"{texto:^40}")
        print(" | ".join(fila))
    print("\n")

def obtener_coeficientes(tabla):
    return [tabla[0][j] for j in range(len(tabla[0])) if tabla[0][j] != ""]

def evaluar_newton(x, coef, x_eval):
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_eval - x[i]) + coef[i]
    return result

# Datos de entrada
x = np.array([-2, -1, 0, 2, 3, 6])
y = np.array([-18, -5, -2, -2, 7, 142])

# Construir pirámides
valores, formulas = construir_piramide(x, y)

print("Pirámide de diferencias divididas - VALORES:")
imprimir_piramide(valores)

print("Pirámide de diferencias divididas - FÓRMULAS:")
imprimir_piramide(formulas)

# Obtener coeficientes
coeficientes = obtener_coeficientes(valores)
print("\nCoeficientes del polinomio de Newton:")
print(np.array(coeficientes))

# Evaluar polinomio
x_val = 5
y_val = evaluar_newton(x, coeficientes, x_val)
print(f"\nInterpolación en x = {x_val}: y = {y_val:.4f}")