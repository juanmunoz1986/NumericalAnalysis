import numpy as np

def verificar_dominancia_diagonal(A_np):
    """
    Verifica si una matriz A_np es estrictamente diagonal dominante por filas.

    Parámetros:
    A_np (np.array): La matriz a verificar.

    Retorna:
    bool: True si es estrictamente diagonal dominante, False en caso contrario.
    """
    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        return False # Solo aplica a matrices cuadradas

    n = A_np.shape[0]
    for i in range(n):
        suma_no_diag = 0
        for j in range(n):
            if i == j:
                continue
            suma_no_diag += abs(A_np[i, j])
        
        if abs(A_np[i, i]) <= suma_no_diag: # Estrictamente dominante requiere >
            return False # No es estrictamente diagonal dominante
    return True # Si todas las filas cumplen

def resolver_sistema_jacobi(A_np, b_np, tol, max_iter, x_inicial_np=None):
    """
    Resuelve un sistema de ecuaciones lineales Ax = b usando el método de Jacobi clásico.

    Parámetros:
    A_np (np.array): Matriz de coeficientes.
    b_np (np.array): Vector de términos independientes.
    tol (float): Tolerancia para la norma del residuo.
    max_iter (int): Número máximo de iteraciones.
    x_inicial_np (np.array, opcional): Estimación inicial para x. Si es None, se usa un vector de ceros.

    Retorna:
    dict: Un diccionario con los resultados:
        {'solucion': np.array, 
         'iteraciones_realizadas': int, 
         'norma_residuo_final': float, 
         'status': str,
         'historial_iteraciones': list of dicts [{'iter': int, 'x_k': np.array, 'norma_residuo': float}]}
    str: Mensaje de error si ocurre un problema de validación inicial.
    """

    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        return "Error: La matriz A debe ser cuadrada."
    if b_np.ndim != 1 or b_np.shape[0] != A_np.shape[0]:
        return "Error: El vector b debe tener la misma dimensión que las filas/columnas de A."

    try:
        detA = np.linalg.det(A_np)
        if np.isclose(detA, 0.0):
            return "Error: La matriz de coeficientes es singular (determinante cercano a cero)."
    except np.linalg.LinAlgError:
        return "Error: No se pudo calcular el determinante de A. Verifique que sea una matriz válida."

    n = A_np.shape[0]
    if x_inicial_np is None:
        x_k = np.zeros(n, dtype=float)
    else:
        if x_inicial_np.shape != (n,):
            return "Error: La estimación inicial x_inicial tiene dimensiones incorrectas."
        x_k = np.copy(x_inicial_np).astype(float)
    
    x_k_siguiente = np.zeros_like(x_k)
    historial_iteraciones = []
    iter_realizadas_count = 0
    norma_residuo_final_calculada = float('inf')

    residuo_inicial = A_np @ x_k - b_np
    norma_residuo_inicial = np.linalg.norm(residuo_inicial)
    historial_iteraciones.append({
        'iter': 0,
        'x_k': np.copy(x_k),
        'norma_residuo': norma_residuo_inicial
    })

    for k_iter_loop in range(1, max_iter + 1):
        iter_realizadas_count = k_iter_loop
        x_k_anterior = np.copy(x_k)

        for i in range(n):
            if np.isclose(A_np[i, i], 0.0):
                return {
                    'solucion': x_k,
                    'iteraciones_realizadas': iter_realizadas_count,
                    'norma_residuo_final': norma_residuo_final_calculada,
                    'status': f"Error: Elemento diagonal A[{i},{i}] es cero. Imposible dividir para Jacobi.",
                    'historial_iteraciones': historial_iteraciones
                }

            sum_axj = 0.0
            for j in range(n):
                if i == j:
                    continue
                sum_axj += A_np[i, j] * x_k_anterior[j]
            
            # Fórmula clásica de Jacobi
            x_k_siguiente[i] = (b_np[i] - sum_axj) / A_np[i, i]
        
        x_k = np.copy(x_k_siguiente)

        residuo_actual_completo = A_np @ x_k - b_np
        norma_residuo_final_calculada = np.linalg.norm(residuo_actual_completo)

        historial_iteraciones.append({
            'iter': k_iter_loop,
            'x_k': np.copy(x_k),
            'norma_residuo': norma_residuo_final_calculada
        })

        if norma_residuo_final_calculada < tol:
            return {
                'solucion': x_k,
                'iteraciones_realizadas': k_iter_loop,
                'norma_residuo_final': norma_residuo_final_calculada,
                'status': "Convergencia alcanzada (Jacobi Clásico).",
                'historial_iteraciones': historial_iteraciones
            }

    return {
        'solucion': x_k,
        'iteraciones_realizadas': iter_realizadas_count,
        'norma_residuo_final': norma_residuo_final_calculada,
        'status': f"No se alcanzó convergencia tras {max_iter} iteraciones (Jacobi Clásico).",
        'historial_iteraciones': historial_iteraciones
    }

if __name__ == '__main__':
    print("Probando el módulo interactive_jacboi.py (Jacobi Clásico)...")
    A_test = np.array([[4.0, -1.0, 1.0],
                       [-1.0, 4.0, -2.0],
                       [1.0, -2.0, 4.0]], dtype=float)
    b_test = np.array([12.0, -1.0, 5.0], dtype=float)
    tol_test = 1e-6
    max_iter_test = 30 

    print("\nCaso 1: Jacobi Clásico")
    resultado_j = resolver_sistema_jacobi(A_test, b_test, tol_test, max_iter_test)
    if isinstance(resultado_j, str):
        print(resultado_j)
    else:
        print(f"Status: {resultado_j['status']}")
        print(f"Solución x final: {np.array2string(resultado_j['solucion'], precision=6)}")
        print(f"Iteraciones realizadas: {resultado_j['iteraciones_realizadas']}")
        print(f"Norma del residuo final: {resultado_j['norma_residuo_final']:.2e}")
        print("Historial (últimas 5 o todas si <=5):")
        hist_j = resultado_j['historial_iteraciones']
        for item in hist_j[-5:]:
            print(f"  Iter {item['iter']:02d}: x_k = {np.array2string(item['x_k'], precision=5, suppress_small=True)}, ||residuo|| = {item['norma_residuo']:.4e}")

    print("\nCaso 2: Matriz del usuario con Jacobi Clásico")
    A_usuario = np.array([[2,   3,   1.5],
                         [1,   2,   0.8],
                         [0.5, 0.7, 1.2]], dtype=float)
    b_usuario = np.array([380, 200, 150], dtype=float)
    resultado_usuario_j = resolver_sistema_jacobi(A_usuario, b_usuario, 1e-6, 200)
    if isinstance(resultado_usuario_j, str):
        print(resultado_usuario_j)
    else:
        print(f"Status: {resultado_usuario_j['status']}")
        print(f"Solución x final: {np.array2string(resultado_usuario_j['solucion'], precision=6)}")
        print(f"Iteraciones realizadas: {resultado_usuario_j['iteraciones_realizadas']}")
        print(f"Norma del residuo final: {resultado_usuario_j['norma_residuo_final']:.2e}")
        print("Historial (últimas 5 iteraciones de la ejecución o todas si son <=5):")
        hist_usr_j = resultado_usuario_j['historial_iteraciones']
        for item in hist_usr_j[-5:]:
             print(f"  Iter {item['iter']:02d}: x_k = {np.array2string(item['x_k'], precision=5, suppress_small=True)}, ||residuo|| = {item['norma_residuo']:.4e}") 