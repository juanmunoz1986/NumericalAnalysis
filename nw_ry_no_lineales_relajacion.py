import numpy as np

def resolver_sistema_newton_raphson(F_func, J_func, x_inicial, tol=1e-6, max_iter=100, w_factor=1.0):
    """
    Resuelve un sistema de ecuaciones no lineales F(x) = 0 usando el método de Newton-Raphson con relajación.

    Parámetros:
    F_func (callable): Función que calcula el vector F(x) de ecuaciones no lineales.
    J_func (callable): Función que calcula la matriz jacobiana J(x) de F.
    x_inicial (np.array): Vector de estimación inicial para las incógnitas.
    tol (float): Tolerancia para la norma del residuo.
    max_iter (int): Número máximo de iteraciones.
    w_factor (float): Factor de relajación (0 < w ≤ 1). 1 = paso completo.

    Retorna:
    dict: Un diccionario con los resultados:
        {'solucion': np.array, 
         'iteraciones_realizadas': int, 
         'norma_residuo_final': float, 
         'status': str,
         'historial_iteraciones': list of dicts [{'iter': int, 'x_k': np.array, 'norma_residuo': float}]}
    """
    if x_inicial is None or not isinstance(x_inicial, np.ndarray):
        return "Error: Se requiere un vector inicial válido."
    
    x = np.copy(x_inicial).astype(float)
    historial_iteraciones = []
    
    # Guardar estado inicial (iteración 0)
    try:
        Fx_inicial = F_func(x)
        norma_inicial = np.linalg.norm(Fx_inicial)
        historial_iteraciones.append({
            'iter': 0,
            'x_k': np.copy(x),
            'Fx_k': np.copy(Fx_inicial),
            'norma_residuo': norma_inicial,
            'Jx_k': None, 
            'delta_k': None,
            'norma_delta_x': None # Nuevo campo para el historial
        })
    except Exception as e:
        return f"Error al evaluar la función F en el punto inicial: {str(e)}"
    
    iter_realizadas_count = 0
    # norma_residuo_final = norma_inicial # Se actualiza en el bucle
    status_final = f"No se alcanzó convergencia tras {max_iter} iteraciones."
    
    for k in range(1, max_iter + 1):
        iter_realizadas_count = k
        x_previo = np.copy(x) # Guardar x_k antes de actualizarlo a x_{k+1}
        
        current_Fx = None
        current_norma_Fx = None
        current_Jx = None
        current_delta_paso = None
        
        try:
            current_Fx = F_func(x) # Esto es F(x_k)
            current_norma_Fx = np.linalg.norm(current_Fx)
        except Exception as e:
            status_final = f"Error al evaluar F en iteración {k}: {str(e)}"
            # Añadir historial parcial antes de salir
            historial_iteraciones.append({
                'iter': k,
                'x_k': np.copy(x_previo), # x_k antes del fallo
                'Fx_k': current_Fx, # Podría ser None si F_func falló
                'norma_residuo': current_norma_Fx, # Podría ser None
                'Jx_k': None,
                'delta_k': None,
                'norma_delta_x': None
            })
            break # Salir del bucle for
        
        # Condición de parada por norma del residuo F(x_k)
        if current_norma_Fx < tol:
            status_final = f"Convergencia alcanzada en {k} iteraciones (norma de F(x_k) < tol)."
            # Registrar esta iteración final antes de salir
            historial_iteraciones.append({
                'iter': k,
                'x_k': np.copy(x_previo), # x_k actual
                'Fx_k': current_Fx,
                'norma_residuo': current_norma_Fx,
                'Jx_k': None, # J y delta no se calcularon si ya convergió por F(x_k)
                'delta_k': None,
                'norma_delta_x': None # No relevante si paró por F(x_k) en este punto
            })
            break # Salir del bucle for

        try:
            current_Jx = J_func(x) # J(x_k)
            current_delta_paso = np.linalg.solve(current_Jx, -current_Fx) # delta_k
        except np.linalg.LinAlgError as e:
            status_final = f"Error: Jacobiano singular en iteración {k}: {str(e)}"
            historial_iteraciones.append({
                'iter': k,
                'x_k': np.copy(x_previo),
                'Fx_k': current_Fx,
                'norma_residuo': current_norma_Fx,
                'Jx_k': current_Jx, # Puede ser el Jacobiano que causó el error
                'delta_k': None,
                'norma_delta_x': None
            })
            break # Salir del bucle for
        except Exception as e:
            status_final = f"Error al calcular el paso en iteración {k}: {str(e)}"
            historial_iteraciones.append({
                'iter': k,
                'x_k': np.copy(x_previo),
                'Fx_k': current_Fx,
                'norma_residuo': current_norma_Fx,
                'Jx_k': current_Jx, # Puede ser None si J_func falló antes
                'delta_k': None,
                'norma_delta_x': None
            })
            break # Salir del bucle for
        
        x = x_previo + w_factor * current_delta_paso # Esto es x_{k+1}
        
        # Calcular norma de la diferencia entre x_k (ahora x) y x_{k-1} (x_previo)
        norma_delta_x_val = np.linalg.norm(x - x_previo, np.inf) if x_previo is not None else None

        # Actualizar el historial con todos los datos de la iteración k (usando x_previo como x_k)
        historial_iteraciones.append({
            'iter': k,
            'x_k': np.copy(x_previo),    # x_k
            'Fx_k': current_Fx,           # Valor de F(x_k)
            'norma_residuo': current_norma_Fx, # Norma de F(x_k)
            'Jx_k': current_Jx,           # Jacobiano J(x_k)
            'delta_k': current_delta_paso, # Paso delta_k calculado
            'norma_delta_x': norma_delta_x_val # Norma ||x_{k+1} - x_k||
        })
        
        # Condición de parada por diferencia entre iteraciones sucesivas de x
        if norma_delta_x_val is not None and norma_delta_x_val < tol:
            status_final = f"Convergencia alcanzada en {k} iteraciones (||x_k - x_{{k-1}}|| < tol)."
            iter_realizadas_count = k # Asegurar que el contador de iteraciones sea correcto
            break # Salir del bucle for
        
        # Si es la última iteración y no ha convergido por ningún criterio aún
        if k == max_iter:
            status_final = f"No se alcanzó convergencia tras {max_iter} iteraciones (ningún criterio cumplido)."
            # La norma del residuo final será la de F(x_previo) si no se calculó una nueva para x
            # O, si queremos ser más precisos, podríamos calcular F(x) aquí una última vez.
            # Por simplicidad, la norma del residuo ya está en current_norma_Fx, que es F(x_k) de esta iteración.
            # x ya es x_{max_iter+1}

    # Determinar la norma del residuo final basándose en el último x calculado
    # Si el bucle terminó antes de max_iter, 'x' es la solución.
    # Si el bucle completó max_iter, 'x' es x_{max_iter+1}.
    final_Fx = F_func(x) # F(solución final) o F(x_{max_iter+1})
    norma_residuo_final_calculada = np.linalg.norm(final_Fx)
    
    return {
        'solucion': x,
        'iteraciones_realizadas': iter_realizadas_count,
        'norma_residuo_final': norma_residuo_final_calculada, # Usar la norma del F(x) final
        'status': status_final,
        'historial_iteraciones': historial_iteraciones
    }

# Funciones predefinidas para el ejemplo
def ejemplo_F_sistema_2x2(vec):
    """
    Ejemplo de sistema no lineal 2x2:
    f1(x,y) = x^2 + xy - 10 = 0
    f2(x,y) = y + 3xy^2 - 57 = 0
    """
    x, y = vec
    return np.array([
        x**2 + x*y - 10,
        y + 3*x*y**2 - 57
    ])

def ejemplo_J_sistema_2x2(vec):
    """
    Jacobiano del sistema de ejemplo:
    J = [∂f1/∂x  ∂f1/∂y]
        [∂f2/∂x  ∂f2/∂y]
    """
    x, y = vec
    return np.array([
        [2*x + y,      x        ],
        [3*y**2,  1 + 6*x*y     ]
    ])

# Código de prueba que se ejecuta solo si se llama directamente a este archivo
if __name__ == '__main__':
    print("Probando el módulo nw_ray_relajacion.py (Newton-Raphson para sistemas no lineales)...")
    
    # Parámetros de prueba
    x_inicial = np.array([1.0, 1.0])
    tol_test = 1e-6
    max_iter_test = 100
    w_test = 1.0  # Sin relajación
    
    print("\nCaso 1: Sistema de ejemplo 2x2")
    resultado = resolver_sistema_newton_raphson(
        ejemplo_F_sistema_2x2, 
        ejemplo_J_sistema_2x2, 
        x_inicial, 
        tol_test, 
        max_iter_test, 
        w_test
    )
    
    if isinstance(resultado, str):
        print(resultado)
    else:
        print(f"Status: {resultado['status']}")
        print(f"Solución x final: {np.array2string(resultado['solucion'], precision=6)}")
        print(f"Iteraciones realizadas: {resultado['iteraciones_realizadas']}")
        print(f"Norma del residuo final: {resultado['norma_residuo_final']:.2e}")
        print("Historial (últimas 5 o todas si <=5):")
        hist = resultado['historial_iteraciones']
        for item in hist[-5:]:
            print(f"  Iter {item['iter']:02d}: x_k = {np.array2string(item['x_k'], precision=5, suppress_small=True)}, ||residuo|| = {item['norma_residuo']:.4e}")
