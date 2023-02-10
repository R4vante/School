import numpy as np

x = np.linspace(0, 1, 10)

y = ((x<=0.2) + (x>0.8)).astype(float)

print(y)