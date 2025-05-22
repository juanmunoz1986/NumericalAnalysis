import numpy as np

def construir_piramide(x, y):
    n = len(x)
    tabla_valores = np.zeros((n, n), dtype=float)
    tabla_valores[:, 0] = y

    tabla_formulas = [["" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        tabla_formulas[i][0] = f"{y[i]:.4f}"  # primera columna es y

    for j in range(1, n):
        for i in range(n - j):
            a = tabla_valores[i + 1][j - 1]
            b = tabla_valores[i][j - 1]
            xa = x[i + j]
            xb = x[i]

            numerador = f"[({a:.2f}) - ({b:.2f})]"
            denominador = f"[({xa}) - ({xb})]"
            formula = f"{numerador} / {denominador}"

            tabla_valores[i][j] = (a - b) / (xa - xb)
            tabla_formulas[i][j] = formula

    return tabla_valores,tabla_formulas

def imprimir_piramide(tabla):
    n = len(tabla)
    for i in range(n):
        fila = [f"{tabla[i][j]:>40} |" for j in range(n+1)]
        print(" ".join(fila))
    print("\n")

def obtener_coeficientes(tabla):
    # Los coeficientes del polinomio de Newton son la primera fila de cada columna
    return [tabla[0][j] for j in range(tabla.shape[1])]

def evaluar_newton(x, coef, x_eval):
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x_eval - x[i]) + coef[i]
    return result

# Datos de entrada
x = np.array([-2, -1, 0, 2, 3, 6])
y = np.array([-18, -5, -2, -2, 7, 142])

# Construir pirámide
valores, formulas = construir_piramide(x, y)

valores = np.hstack((x.reshape(-1,1),valores))
formulas = np.hstack((x.reshape(-1,1), formulas))

print("Pirámide de diferencias divididas valores:")
imprimir_piramide(valores)


print("Pirámide de diferencias divididas formulas:")
imprimir_piramide(formulas)

# Obtener coeficientes desde la pirámide
coeficientes = obtener_coeficientes(valores)
print("\nCoeficientes del polinomio de Newton:")
print(coeficientes)

# Evaluar el polinomio en un punto
x_val = 5
y_val = evaluar_newton(x, coeficientes, x_val)
print(f"\nInterpolación en x = {x_val}: y = {y_val:.4f}")
