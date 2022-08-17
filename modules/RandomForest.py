import matplotlib.pyplot as plt 
import numpy as np 
x = np.arange(3) 
y1 = [218, 220,188] 
y2 = [122, 119,110] 
width = 0.40
plt.bar(x+0.2, y1, width) 
plt.bar(x-0.2, y2, width) 
plt.xticks(x, ['Aritmética', 'Escrita', 'Leitura']) 
plt.legend(["Antes de Usar ACO", "Depois de Usar ACO"])
plt.show()