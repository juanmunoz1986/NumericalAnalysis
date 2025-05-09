
import numpy as np                  # Librería de cálculo numérico
import sys                          # Para finalizar con códigos de error

# ───────────────────────────────────────────────────────────────
# 1. Definimos las funciones no lineales F y su jacobiano J
# ───────────────────────────────────────────────────────────────
def F(vec):
    """
    Calcula el vector F(x) = [f1(x,y), f2(x,y)].
    Parámetros
    ----------
    vec : ndarray shape (2,)
          Contiene las variables [x, y].
    Retorna
    -------
    ndarray shape (2,)
    """
    x, y = vec                      # Desempaquetamos x e y
    return np.array([
        x**2 + x*y - 10,            # f1(x, y)
        y + 3*x*y**2 - 57           # f2(x, y)
    ])

def J(vec):
    """
    Calcula la matriz jacobiana J(x) de tamaño 2×2.
    J_ij = ∂f_i / ∂x_j
    """
    x, y = vec
    return np.array([
        [2*x + y,      x        ],  # ∂f1/∂x , ∂f1/∂y
        [3*y**2,  1 + 6*x*y     ]   # ∂f2/∂x , ∂f2/∂y
    ])

# ───────────────────────────────────────────────────────────────
# 2. Parámetros del algoritmo NR con relajación
# ───────────────────────────────────────────────────────────────
w           = 1      # Factor de relajación (0<w≤1). 1 = paso completo
tol         = 0.000001    # Tolerancia de convergencia en ||F||
max_iter    = 100      # Iteraciones máximas permitidas
mostrar_pas = True     # Imprimir cada iteración

# ───────────────────────────────────────────────────────────────
# 3. Estimación inicial (elige otra si lo deseas)
# ───────────────────────────────────────────────────────────────
x = np.array([1, 1])   # Vector [x0, y0]

# ───────────────────────────────────────────────────────────────
# 4. Bucle principal de Newton–Raphson con relajación
# ───────────────────────────────────────────────────────────────
for k in range(1, max_iter + 1):
    Fx    = F(x)                           # Vector función en x_k
    norma = np.linalg.norm(Fx)             # Norma L2 del residuo

    if mostrar_pas:
        print(f"Iter {k:02d}  ||F|| = {norma:.3e}   x = {x}")

    if norma < tol:                        # ¿Convergencia?
        print("\n✅ Convergencia alcanzada.\n")
        break

    Jx = J(x)                              # Jacobiano en x_k
    try:
        delta = np.linalg.solve(Jx, -Fx)   # Resolvemos J·Δ = –F
    except np.linalg.LinAlgError as e:     # Jacobiano singular
        print(f"❌ Jacobiano singular en la iteración {k}: {e}")
        sys.exit(1)

    x = x + w * delta                      # Actualizamos con relajación

else:
    # Salimos del for sin converger
    print(f"❌ No se alcanzó convergencia tras {max_iter} iteraciones "
          f"(||F|| = {norma:.3e}).")
    sys.exit(2)

# ───────────────────────────────────────────────────────────────
# 5. Resultado final
# ───────────────────────────────────────────────────────────────
print("Resultado final:")
print(f"x = {x[0]:.10f}")
print(f"y = {x[1]:.10f}")
