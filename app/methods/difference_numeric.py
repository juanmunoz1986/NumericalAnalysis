#sucesiva
#centralizada
#regresiva
import numpy as np

#Condición, igualmente espaciado entre valores de x

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])


#Evaluar si es igualmente espaciado.

def is_spacing_equal(x):
    spacing_value = x[1] - x[0]
    for x_i in range(1, len(x)):
        if x[x_i] - x[x_i - 1] != spacing_value:
            return False
    return True


print("Es igualmente espaciado entre los valores de x: ", is_spacing_equal(x))
