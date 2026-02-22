# tiempos entre latidos (en segundos)
beats = [0.82, 0.84, 0.81, 0.86, 0.83, 0.80] # lista de tiempos entre latidos

import numpy as np

beats_np = np.array(beats) # convertir la lista a un array de numpy para facilitar los cálculos

print("Promedio:", beats_np.mean()) # calcula el promedio de los tiempos entre latidos
print("Variación:", beats_np.std()) # la desviación estándar es una medida de la variación de los tiempos entre latidos

import matplotlib.pyplot as plt 


plt.plot(beats_np) # plot agarra el valor de cada latido y lo usa en el eje y, y la posicion del latido en la lista (0, 1, 2, etc.) se usa en el eje x
plt.title("Ritmo cardíaco")
plt.xlabel("Latido")
plt.ylabel("Segundos")
plt.show()