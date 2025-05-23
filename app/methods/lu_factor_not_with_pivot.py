import numpy as np
import scipy.linalg as la

def lu_no_pivot(A):
    """
    Realiza factorización LU sin pivoteo: A = L * U
    Retorna L y U
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)

    for i in range(n):
        if np.isclose(U[i, i], 0):
            raise ZeroDivisionError("Pivote nulo. LU sin pivoteo no puede continuar.")
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, :] -= factor * U[i, :]
    return L, U


def resolver_sistema_lu(A_np, b_np, usar_pivoteo=False):
    """
    Resuelve un sistema de ecuaciones lineales Ax = b usando factorización LU.

    Parámetros:
    A_np (np.array): Matriz de coeficientes.
    b_np (np.array): Vector de términos independientes.
    usar_pivoteo (bool): Si es True, usa scipy.linalg.lu (con pivoteo).
                         Si es False, usa lu_no_pivot (sin pivoteo).

    Retorna:
    tuple: (P, L, U, x, y, b_modificado)
    """
    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        return "Error: la matriz A debe ser cuadrada."
    if b_np.ndim != 1 or b_np.shape[0] != A_np.shape[0]:
        return "Error: el vector b debe tener la misma dimensión que las filas/columnas de A."

    try:
        detA = np.linalg.det(A_np)
        if abs(detA) < 1e-12:
            return "Error: la matriz A es singular (determinante cercano a cero)."
    except np.linalg.LinAlgError:
        return "Error: no se pudo calcular el determinante de A."

    try:
        if usar_pivoteo:
            # Usa scipy con pivoteo
            P, L, U = la.lu(A_np)
            b_mod = P @ b_np
        else:
            # Usa implementación sin pivoteo
            L, U = lu_no_pivot(A_np)
            b_mod = b_np  # No se aplica permutación
            P = np.eye(A_np.shape[0])  # Devolvemos la identidad por compatibilidad
    except Exception as e:
        return f"Error durante la factorización LU: {str(e)}"

    # Resolver LY = b_mod (hacia adelante)
    try:
        y = la.solve_triangular(L, b_mod, lower=True, unit_diagonal=True)
    except Exception as e:
        return f"Error durante la sustitución hacia adelante: {str(e)}"

    # Resolver UX = y (hacia atrás)
    try:
        x = la.solve_triangular(U, y, lower=False)
    except Exception as e:
        return f"Error durante la sustitución hacia atrás: {str(e)}"

    return P, L, U, x, y, b_mod


# Código de prueba
if __name__ == '__main__':
    print("Probando el módulo con y sin pivoteo...\n")

    A_test = np.array([[2, -3, 1],
                       [-4, 9,  2],
                       [6, -12,  -2]], dtype=float)
    b_test = np.array([3, 4, -2], dtype=float)

    print("Caso 1: Sin pivoteo (esperado L y U específicos)")
    resultado = resolver_sistema_lu(A_test, b_test, usar_pivoteo=False)
    if isinstance(resultado, str):
        print(resultado)
    else:
        P_res, L_res, U_res, x_res, y_res, b_mod = resultado
        print("Matriz L:\n", L_res)
        print("Matriz U:\n", U_res)
        print("Vector Y (LY = b):\n", y_res)
        print("Vector X (UX = Y):\n", x_res)

    print("\nCaso 2: Con pivoteo (para comparación)")
    resultado_pivot = resolver_sistema_lu(A_test, b_test, usar_pivoteo=True)
    if isinstance(resultado_pivot, str):
        print(resultado_pivot)
    else:
        P_res, L_res, U_res, x_res, y_res, b_mod = resultado_pivot
        print("Matriz P:\n", P_res)
        print("Matriz L:\n", L_res)
        print("Matriz U:\n", U_res)
        print("Vector Y:\n", y_res)
        print("Vector X:\n", x_res)
