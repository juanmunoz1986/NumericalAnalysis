# Proyecto de Métodos Numéricos con Interfaz Gráfica (Tkinter)

Este proyecto académico tiene como objetivo ofrecer una herramienta interactiva y educativa que permita visualizar y aplicar distintos **métodos numéricos** para la solución de **sistemas de ecuaciones lineales y no lineales**, a través de una interfaz gráfica desarrollada en **Tkinter**.

## 🎯 Objetivos

* Implementar y comparar métodos numéricos clásicos utilizados en álgebra lineal y análisis numérico.
* Desarrollar una interfaz gráfica amigable que facilite el uso de estos métodos sin requerir conocimientos avanzados de programación.
* Fomentar la comprensión visual de la convergencia y comportamiento de los métodos iterativos.
* Servir como herramienta educativa para estudiantes y docentes de ingeniería o ciencias afines.

---

## 📦 Estructura del Proyecto

```
E:.
│   main.py                 → Archivo principal para ejecutar la GUI
│   requirements            → Dependencias del proyecto
│   README.md               → Este archivo
│
├───app
│   │   __init__.py
│   │
│   ├───gui
│   │   main_window.py      → Ventana principal y lógica de la interfaz Tkinter
│   │   __init__.py
│   │
│   ├───methods
│   │   ├───interpolation
│   │   │   finite_differences.py         → Método de diferencias finitas
│   │   │   newton_divided_differences.py → Método de Newton por diferencias divididas
│   │   │   __init__.py
│   │   │
│   │   ├───root_finding
│   │   │   interactive_jacboi.py                → Método de Jacobi iterativo
│   │   │   lu_factorization_no_pivot.py         → Factorización LU sin pivoteo
│   │   │   lu_factorization_with_pivot.py       → Factorización LU con pivoteo parcial
│   │   │   newton_raphson_no_line_relaxation.py → Newton-Raphson sin relajación
│   │   │   newton_raphson_with_relaxation.py    → Newton-Raphson con relajación
│   │   │   __init__.py
│   │
│   ├───utils
│   │   __init__.py         → Funciones auxiliares (si las hay)
│
└───.idea                   → Configuración del entorno de desarrollo (PyCharm)
```

---

## 🧮 Métodos Numéricos Implementados

### 📌 Sistemas de Ecuaciones Lineales

1. **Factorización LU (con/sin pivoteo)**

   * Descompone la matriz A en L y U para resolver Ax = b.
   * Incluye visualización de matrices P, L, U y solución final x.

2. **Método de Jacobi**

   * Iterativo, requiere condición de dominancia diagonal.
   * Muestra historial completo de iteraciones.

3. **Método de Gauss-Seidel y SOR (relajación)**

   * Versión iterativa mejorada.
   * Permite modificar el parámetro de relajación w.
   * Muestra evolución paso a paso.

### 📌 Sistemas de Ecuaciones No Lineales

4. **Método de Newton-Raphson para sistemas 2x2**

   * Permite ingresar funciones F1(x, y), F2(x, y) y la matriz Jacobiana.
   * Visualiza curvas de nivel y trayectoria de convergencia.
   * Muestra resumen y detalles de iteraciones (normas, residuo, etc.).

### 📌 Comparación de Métodos

5. **Comparador de métodos lineales**

   * Resuelve el mismo sistema usando LU, Jacobi y Gauss-Seidel/SOR.
   * Visualiza resultados y, si es posible (2x2 o 3x3), grafica las soluciones.

---

## 🖥️ Requisitos

* Python 3.x
* Bibliotecas necesarias:

  * `numpy`
  * `matplotlib`
  * `tkinter` (incluida por defecto en la mayoría de instalaciones de Python)

Instalación rápida:

```bash
pip install numpy matplotlib
```

---

## 🚀 Cómo Ejecutar la Aplicación

1. Clona este repositorio o descarga los archivos:

   ```bash
   git clone https://github.com/juanmunoz1986/numericalanalysis.git
   ```

2. Asegúrate de tener Python instalado y las bibliotecas requeridas.

3. Abre una terminal o CMD, navega a la carpeta del proyecto y ejecuta:

   ```bash
   python main.py
   ```

4. Se abrirá una ventana gráfica donde podrás seleccionar y utilizar los métodos numéricos.

---

## 🧠 Uso del Proyecto

* Cada método cuenta con un formulario donde el usuario puede ingresar los datos requeridos (matriz A, vector b, tolerancia, iteraciones, etc.).
* Al ejecutar el método, se muestran los resultados detallados y gráficos, si aplica.
* La interfaz fue diseñada para ser intuitiva y flexible, facilitando tanto la exploración como el aprendizaje de los métodos.

---

## 👨‍💻 Créditos

Este proyecto fue desarrollado como parte del curso de **Métodos Numéricos** con fines académicos, para aplicar conocimientos en Python, análisis numérico y diseño de interfaces gráficas.
