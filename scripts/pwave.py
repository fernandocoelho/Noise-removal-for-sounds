import numpy as np
import matplotlib.pyplot as plt

i = raw_input()
a = i.split(" ")
a.pop()
a = map(int, a)
plt.plot(np.arange(len(a)), a, "r-")
#i = raw_input()
#a = i.split(" ")
#a.pop()
#a = map(int, a)
#plt.plot(np.arange(len(a)), a, "b:")
plt.show()
