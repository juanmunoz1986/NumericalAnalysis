# Proyecto de Métodos Numéricos con Interfaz Gráfica

Este proyecto implementa varios métodos numéricos para resolver sistemas de ecuaciones lineales y no lineales, junto con una interfaz gráfica de usuario (GUI) construida con Tkinter en Python.

## Características

El proyecto permite a los usuarios ingresar los parámetros necesarios para cada método y visualiza los resultados, incluyendo, en algunos casos, representaciones gráficas de la convergencia o del sistema de ecuaciones.

### Métodos Implementados:

1.  **Factorización LU (Sistemas Lineales)**:
    *   Resuelve sistemas de ecuaciones lineales Ax = b mediante la descomposición LU de la matriz A (con pivotaje parcial P).
    *   Muestra las matrices P, L, U y el vector solución x.

2.  **Gauss-Seidel / SOR (Sistemas Lineales Iterativo)**:
    *   Resuelve sistemas lineales Ax = b utilizando el método iterativo de Gauss-Seidel o su variante con Sobre-Relajación Sucesiva (SOR).
    *   El usuario puede especificar la matriz A, el vector b, el vector inicial x0, el factor de relajación w (para SOR), la tolerancia y el número máximo de iteraciones.
    *   Muestra el historial de iteraciones.

3.  **Jacobi Clásico (Sistemas Lineales Iterativo)**:
    *   Resuelve sistemas lineales Ax = b utilizando el método iterativo de Jacobi.
    *   El usuario puede especificar la matriz A, el vector b, el vector inicial x0, la tolerancia y el número máximo de iteraciones.
    *   Verifica la dominancia diagonal de la matriz A.
    *   Muestra el historial de iteraciones.

4.  **Newton-Raphson (Sistemas No Lineales)**:
    *   Resuelve sistemas de dos ecuaciones no lineales F(x) = 0.
    *   Permite al usuario definir las funciones F1(x,y), F2(x,y) y las componentes de su matriz Jacobiana J(x,y) como expresiones de texto.
    *   También incluye un sistema de ejemplo predefinido.
    *   El usuario especifica el vector inicial, la tolerancia, el máximo de iteraciones y un factor de relajación.
    *   Muestra un resumen de la solución (estado, solución, iteraciones, norma del residuo).
    *   Muestra un historial detallado de cada iteración (x_k, F(x_k), J(x_k), delta_k, norma del residuo, norma del cambio en x).
    *   Proporciona una visualización gráfica de las curvas de nivel F1=0, F2=0 y la trayectoria de convergencia de la solución.

5.  **Comparación de Métodos Lineales**:
    *   Permite ingresar un sistema lineal y resolverlo simultáneamente por Factorización LU, Jacobi y Gauss-Seidel/SOR.
    *   Muestra los resultados de cada método en pestañas separadas.
    *   Incluye una pestaña de visualización que, para sistemas 2x2, grafica las dos ecuaciones y la solución, y para sistemas 3x3, intenta graficar los planos y la solución.

## Requisitos

*   Python 3.x
*   NumPy
*   Matplotlib
*   Tkinter (generalmente incluido con las instalaciones estándar de Python)

Se recomienda instalar las bibliotecas necesarias usando pip:
```bash
pip install numpy matplotlib
```

## Cómo Ejecutar el Proyecto

1.  Asegúrate de tener Python y las bibliotecas requeridas instaladas.
2.  Clona este repositorio o descarga los archivos del proyecto en una carpeta local.
3.  Navega a la carpeta del proyecto en tu terminal.
4.  Ejecuta el archivo principal de la interfaz gráfica:

    ```bash
    python main_gui.py
    ```

Esto abrirá la ventana principal de la aplicación, desde donde podrás seleccionar y utilizar los diferentes métodos numéricos.

## Estructura de Archivos

*   `main_gui.py`: Contiene la lógica de la interfaz gráfica principal de Tkinter y la coordinación entre la GUI y los módulos de los métodos numéricos.
*   `fac_LU.py`: Implementación del método de Factorización LU.
*   `jacobi_method.py`: Implementación del método de Jacobi.
*   `nw_ray_relajacion.py`: Implementación del método de Gauss-Seidel / SOR.
*   `nw_ry_no_lineales_relajacion.py`: Implementación del método de Newton-Raphson para sistemas no lineales.
*   `README.md`: Este archivo.

## Contribuciones

Este proyecto fue desarrollado con el objetivo de proporcionar herramientas educativas para la comprensión de métodos numéricos. 