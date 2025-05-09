import numpy as np                 # Importamos NumPy para manejo de arrays y operaciones numéricas
import scipy.linalg as la          # Importamos SciPy (linalg) para utilizar la factorización LU

def resolver_sistema_lu(A_np, b_np):
    """
    Resuelve un sistema de ecuaciones lineales Ax = b usando factorización LU.

    Parámetros:
    A_np (np.array): Matriz de coeficientes.
    b_np (np.array): Vector de términos independientes.

    Retorna:
    tuple: (P, L, U, x) si la solución es exitosa.
    str: Mensaje de error si ocurre un problema.
    """

    # 5. Manejo de errores: verificar que A sea cuadrada y que las dimensiones coincidan con b.
    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        return "Error: la matriz A debe ser cuadrada."
    if b_np.ndim != 1 or b_np.shape[0] != A_np.shape[0]:
        return "Error: el vector b debe tener la misma dimensión que las filas/columnas de A."

    # Verificar que la matriz no sea singular.
    try:
        # Intentamos calcular el determinante. Si es muy cercano a cero, la matriz es singular.
        # Usamos try-except porque np.linalg.det puede fallar para matrices no cuadradas (aunque ya lo validamos antes)
        detA = np.linalg.det(A_np)
        if abs(detA) < 1e-12: # Umbral más estricto puede ser necesario dependiendo de la escala de los números
            return "Error: la matriz A es singular (determinante cercano a cero) y no se puede factorizar LU de forma estable."
    except np.linalg.LinAlgError:
        # Esto podría ocurrir si la matriz no es cuadrada o tiene otros problemas.
        return "Error: No se pudo calcular el determinante de A. Verifique que sea una matriz válida."


    # 2. Factorización LU de la matriz A.
    try:
        P, L, U = la.lu(A_np)
    except np.linalg.LinAlgError as e:
        return f"Error durante la factorización LU: {str(e)}"

    # 3. Sustitución hacia adelante: resolver L * y = P * b.
    # El resultado de P.dot(b_np) es b_permutado
    try:
        b_permutado = P @ b_np # Usamos @ para dot product en versiones recientes de numpy
        y = la.solve_triangular(L, b_permutado, lower=True, unit_diagonal=False) # L puede no tener 1s en la diagonal con la.lu
    except np.linalg.LinAlgError as e:
        return f"Error durante la sustitución hacia adelante: {str(e)}"
    except ValueError as e: # por si las dimensiones no calzan por alguna razón inesperada
        return f"Error de dimensiones en sustitución hacia adelante: {str(e)}"

    # 4. Sustitución hacia atrás: resolver U * x = y.
    try:
        x = la.solve_triangular(U, y, lower=False)
    except np.linalg.LinAlgError as e:
        return f"Error durante la sustitución hacia atrás: {str(e)}"
    except ValueError as e:
        return f"Error de dimensiones en sustitución hacia atrás: {str(e)}"

    return P, L, U, x

# El siguiente código es para probar el módulo directamente si se ejecuta este archivo.
# No se ejecutará cuando se importe desde main_gui.py
if __name__ == '__main__':
    print("Probando el módulo fac_LU.py...")
    # Ejemplo de uso:
    A_test = np.array([[2, -3, 1],
                       [-4, 9,  2],
                       [6, -12,  -2]], dtype=float)
    b_test = np.array([3, 4, -2], dtype=float)

    print("\nCaso 1: Matriz válida")
    resultado = resolver_sistema_lu(A_test, b_test)
    if isinstance(resultado, str):
        print(resultado)  # Imprime el mensaje de error
    else:
        P_res, L_res, U_res, x_res = resultado
        print("Matriz P (Permutación):\n", P_res)
        print("Matriz L (Triangular Inferior):\n", L_res)
        print("Matriz U (Triangular Superior):\n", U_res)
        print("Vector x (Solución):\n", x_res)
        # Verificación opcional: A @ x debe ser cercano a b
        # print("Verificación A @ x = ", A_test @ x_res)
        # print("Vector b original = ", b_test)

    print("\nCaso 2: Matriz singular")
    A_singular = np.array([[1, 1], [1, 1]], dtype=float)
    b_singular = np.array([2, 3], dtype=float)
    resultado_singular = resolver_sistema_lu(A_singular, b_singular)
    print(resultado_singular)

    print("\nCaso 3: Matriz no cuadrada")
    A_no_cuadrada = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    b_no_cuadrada = np.array([1,2], dtype=float)
    resultado_no_cuadrada = resolver_sistema_lu(A_no_cuadrada, b_no_cuadrada)
    print(resultado_no_cuadrada)

    print("\nCaso 4: Dimensiones b incorrectas")
    b_dim_mal = np.array([1,2,3], dtype=float)
    resultado_b_dim_mal = resolver_sistema_lu(A_test, b_dim_mal) # Usando A_test que es 3x3
    print(resultado_b_dim_mal)
