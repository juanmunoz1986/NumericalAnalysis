import numpy as np
# sys ya no es necesario aquí, los errores/estados se retornan

def resolver_sistema_sor(A_np, b_np, w_factor, tol, max_iter, x_inicial_np=None):
    """
    Resuelve un sistema de ecuaciones lineales Ax = b usando el método de
    Sobrerrelajación Sucesiva (SOR) o Gauss-Seidel si w_factor = 1.

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
    # if not (0 < w_factor < 2): # Podríamos ser menos estrictos o dejar al usuario experimentar
    #     return "Advertencia: El factor de relajación w para SOR usualmente está en (0, 2)."

    try:
        detA = np.linalg.det(A_np)
        if np.isclose(detA, 0.0):
            return "Error: La matriz de coeficientes es singular (determinante cercano a cero)."
    except np.linalg.LinAlgError:
        return "Error: No se pudo calcular el determinante de A. Verifique que sea una matriz válida."

    n = A_np.shape[0]
    if x_inicial_np is None:
        x_k = np.zeros(n, dtype=float) # Asegurar que sea float para los cálculos
    else:
        if x_inicial_np.shape != (n,):
            return "Error: La estimación inicial x_inicial tiene dimensiones incorrectas."
        x_k = np.copy(x_inicial_np).astype(float)
    
    historial_iteraciones = []
    iter_realizadas_count = 0
    norma_residuo_final_calculada = float('inf')

    # Guardar estado inicial (iteración 0)
    residuo_inicial = A_np @ x_k - b_np
    norma_residuo_inicial = np.linalg.norm(residuo_inicial)
    historial_iteraciones.append({
        'iter': 0,
        'x_k': np.copy(x_k),
        'norma_residuo': norma_residuo_inicial
    })
    
    # Si la estimación inicial ya es buena, podríamos parar.
    # Sin embargo, el bucle realizará al menos una iteración de cálculo.

    for k_iter_loop in range(1, max_iter + 1):
        iter_realizadas_count = k_iter_loop
        x_k_anterior = np.copy(x_k) # x de la iteración (k)

        for i in range(n): # Para cada componente de x
            if np.isclose(A_np[i, i], 0.0):
                # Añadir estado actual al historial antes de fallar
                residuo_actual = A_np @ x_k - b_np # Usar x_k parcialmente actualizado
                norma_r_actual = np.linalg.norm(residuo_actual)
                # No se añade al historial principal de iteraciones completas, sino que se retorna error.
                return {
                    'solucion': x_k,
                    'iteraciones_realizadas': iter_realizadas_count,
                    'norma_residuo_final': norma_r_actual,
                    'status': f"Error: Elemento diagonal A[{i},{i}] es cero en iteración {k_iter_loop}. Imposible dividir.",
                    'historial_iteraciones': historial_iteraciones # Solo hasta la iteración anterior completa
                }

            sum_j_menor_i = 0.0
            for j in range(i):
                sum_j_menor_i += A_np[i, j] * x_k[j]  # Usa x_k[j] ya actualizado en esta iteración (k+1)
            
            sum_j_mayor_i = 0.0
            for j in range(i + 1, n):
                sum_j_mayor_i += A_np[i, j] * x_k_anterior[j] # Usa x_k_anterior[j] de la iteración (k)

            # x_i sin relajación (término de Gauss-Seidel puro)
            x_i_gs = (b_np[i] - sum_j_menor_i - sum_j_mayor_i) / A_np[i, i]
            
            # Aplicar relajación SOR
            x_k[i] = (1 - w_factor) * x_k_anterior[i] + w_factor * x_i_gs

        # Después de actualizar todas las componentes de x_k para la iteración k_iter_loop
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
                'status': "Convergencia alcanzada.",
                'historial_iteraciones': historial_iteraciones
            }

    # Si se alcanza este punto, no hubo convergencia en max_iter
    return {
        'solucion': x_k,
        'iteraciones_realizadas': iter_realizadas_count,
        'norma_residuo_final': norma_residuo_final_calculada,
        'status': f"No se alcanzó convergencia tras {max_iter} iteraciones.",
        'historial_iteraciones': historial_iteraciones
    }

if __name__ == '__main__':
    print("Probando el módulo newton_raphson_with_relaxation.py (Gauss-Seidel/SOR)...")
    # Matriz diagonalmente dominante para asegurar convergencia de Gauss-Seidel
    A_test = np.array([[4.0, -1.0, 1.0],
                       [-1.0, 4.0, -2.0],
                       [1.0, -2.0, 4.0]], dtype=float)
    b_test = np.array([12.0, -1.0, 5.0], dtype=float)
    tol_test = 1e-6
    max_iter_test = 30 # Reducido para no tener salida muy larga

    print("\nCaso 1: Gauss-Seidel (w=1.0)")
    resultado_gs = resolver_sistema_sor(A_test, b_test, 1.0, tol_test, max_iter_test)
    if isinstance(resultado_gs, str):
        print(resultado_gs)
    else:
        print(f"Status: {resultado_gs['status']}")
        print(f"Solución x final: {np.array2string(resultado_gs['solucion'], precision=6)}")
        print(f"Iteraciones realizadas: {resultado_gs['iteraciones_realizadas']}")
        print(f"Norma del residuo final: {resultado_gs['norma_residuo_final']:.2e}")
        print("Historial (últimas 5 o todas si <=5):")
        hist_gs = resultado_gs['historial_iteraciones']
        for item in hist_gs[-5:]:
            print(f"  Iter {item['iter']:02d}: x_k = {np.array2string(item['x_k'], precision=5, suppress_small=True)}, ||residuo|| = {item['norma_residuo']:.4e}")

    print("\nCaso 2: SOR (w=1.2)")
    # Para la matriz de prueba, w=1.2 podría ser un buen factor de relajación
    resultado_sor_w = resolver_sistema_sor(A_test, b_test, 1.2, tol_test, max_iter_test)
    if isinstance(resultado_sor_w, str):
        print(resultado_sor_w)
    else:
        print(f"Status: {resultado_sor_w['status']}")
        print(f"Solución x final: {np.array2string(resultado_sor_w['solucion'], precision=6)}")
        print(f"Iteraciones realizadas: {resultado_sor_w['iteraciones_realizadas']}")
        print(f"Norma del residuo final: {resultado_sor_w['norma_residuo_final']:.2e}")
        print("Historial (últimas 5 o todas si <=5):")
        hist_sor = resultado_sor_w['historial_iteraciones']
        for item in hist_sor[-5:]:
             print(f"  Iter {item['iter']:02d}: x_k = {np.array2string(item['x_k'], precision=5, suppress_small=True)}, ||residuo|| = {item['norma_residuo']:.4e}")

    print("\nCaso 3: Elemento diagonal cero")
    A_diag_cero = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=float)
    b_diag_cero = np.array([1.0, 2.0], dtype=float)
    resultado_diag_cero = resolver_sistema_sor(A_diag_cero, b_diag_cero, 1.0, 1e-5, 10)
    if isinstance(resultado_diag_cero, str):
        print(resultado_diag_cero) # Debería ser un error directo de validación
    else:
        print(f"Status: {resultado_diag_cero['status']}") # O un error de status en el dict
        if 'solucion' in resultado_diag_cero:
             print(f"Solución (parcial): {np.array2string(resultado_diag_cero['solucion'], precision=6)}")

    print("\nCaso 4: Matriz del usuario con w=1 (Gauss-Seidel)")
    A_usuario = np.array([[2,   3,   1.5],
                         [1,   2,   0.8],
                         [0.5, 0.7, 1.2]], dtype=float)
    b_usuario = np.array([380, 200, 150], dtype=float)
    resultado_usuario_gs = resolver_sistema_sor(A_usuario, b_usuario, 1.0, 1e-6, 100)
    if isinstance(resultado_usuario_gs, str):
        print(resultado_usuario_gs)
    else:
        print(f"Status: {resultado_usuario_gs['status']}")
        print(f"Solución x final: {np.array2string(resultado_usuario_gs['solucion'], precision=6)}")
        print(f"Iteraciones realizadas: {resultado_usuario_gs['iteraciones_realizadas']}")
        print(f"Norma del residuo final: {resultado_usuario_gs['norma_residuo_final']:.2e}")
        print("Historial (últimas 5 iteraciones de la ejecución o todas si son <=5):")
        hist_usr_gs = resultado_usuario_gs['historial_iteraciones']
        for item in hist_usr_gs[-5:]:
             print(f"  Iter {item['iter']:02d}: x_k = {np.array2string(item['x_k'], precision=5, suppress_small=True)}, ||residuo|| = {item['norma_residuo']:.4e}")

    print("\nCaso 5: Matriz del usuario con w=1.270 (SOR)")
    resultado_usuario_sor = resolver_sistema_sor(A_usuario, b_usuario, 1.270, 1e-6, 100)
    if isinstance(resultado_usuario_sor, str):
        print(resultado_usuario_sor)
    else:
        print(f"Status: {resultado_usuario_sor['status']}")
        print(f"Solución x final: {np.array2string(resultado_usuario_sor['solucion'], precision=6)}")
        print(f"Iteraciones realizadas: {resultado_usuario_sor['iteraciones_realizadas']}")
        print(f"Norma del residuo final: {resultado_usuario_sor['norma_residuo_final']:.2e}")
        print("Historial (últimas 5 iteraciones de la ejecución o todas si son <=5):")
        hist_usr_sor = resultado_usuario_sor['historial_iteraciones']
        for item in hist_usr_sor[-5:]:
             print(f"  Iter {item['iter']:02d}: x_k = {np.array2string(item['x_k'], precision=5, suppress_small=True)}, ||residuo|| = {item['norma_residuo']:.4e}")
