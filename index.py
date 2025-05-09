import numpy as np                  # Importamos NumPy para trabajar con arreglos y operaciones numéricas
import scipy.linalg as la           # Importamos de SciPy el módulo linalg para usar la factorización LU
import sys                          # Importamos sys para manejar posibles errores y terminar el programa

# ==============================
# DEFINICIÓN DE MATRIZ 3x3 Y VECTOR
# ==============================
# Puedes cambiar estos valores por cualquier sistema compatible 3x3
A = np.array([[2, -3, 1],      # Matriz de coeficientes A (4x4 en este ejemplo)
              [-4, 9,  2],
              [6, -12,  -2]
              ], dtype=float)
b = np.array([3, 4, -2], dtype=float)  # Vector de términos independientes b (dimension 4)

# ==============================
# VALIDACIÓN DE LA MATRIZ
# ==============================
if A.shape[0] != A.shape[1]:            # Verificamos si A no es cuadrada
    print("Error: la matriz A debe ser cuadrada.")
    sys.exit(1)

if b.shape[0] != A.shape[0]:            # Verificamos si b tiene la misma dimensión que A
    print("Error: el vector b debe tener el mismo número de filas que A.")
    sys.exit(1)

if abs(np.linalg.det(A)) < 1e-9:        # Verificamos si el determinante es muy cercano a cero (matriz singular)
    print("Error: la matriz A es singular, no se puede aplicar factorización LU.")
    sys.exit(1)

# ==============================
# FACTORIZACIÓN LU
# ==============================
P, L, U = la.lu(A)                     # Factorizamos: P*A = L*U

print("Matriz L (triangular inferior):")
print(L)                              # Mostramos la matriz L

print("Matriz U (triangular superior):")
print(U)                              # Mostramos la matriz U

# ==============================
# SUSTITUCIÓN HACIA ADELANTE: L·y = P·b
# ==============================
b_permutado = P.dot(b)                # Aplicamos la permutación al vector b
n = A.shape[0]                        # Número de incógnitas
y = np.zeros(n)                       # Inicializamos vector y con ceros

for i in range(n):                    # Para cada fila i
    suma = sum(L[i, j] * y[j] for j in range(i))     # Sumamos los productos L[i,j] * y[j] (j < i)
    y[i] = (b_permutado[i] - suma) / L[i, i]         # Despejamos y[i]

print("Vector y (solución de L·y = P·b):")
print(y)

# ==============================
# SUSTITUCIÓN HACIA ATRÁS: U·x = y
# ==============================
x = np.zeros(n)                       # Inicializamos vector x con ceros

for i in range(n-1, -1, -1):          # Recorremos las filas desde la última hacia la primera
    suma = sum(U[i, j] * x[j] for j in range(i+1, n))  # Sumamos productos U[i,j] * x[j] (j > i)
    x[i] = (y[i] - suma) / U[i, i]                        # Despejamos x[i]

print("Vector x (solución de A·x = b):")
print(x)
