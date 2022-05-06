import numpy as np
import matplotlib.pyplot as plt


np.seterr(invalid='ignore', over='ignore')  # suppress warning caused by division by inf

def f(x):
    return 1/(1 + np.exp(3*(x-3))) * 10 * x**2  + 1 / (1 + np.exp(-3*(x-3))) * (0.5*(x-10)**2 + 50)

def fprime(x):
    return 1 / (1 + np.exp((-3)*(x-3))) * (x-10) + 1/(1 + np.exp(3*(x-3))) * 20 * x + (3* np.exp(9))/(np.exp(9-1.5*x) + np.exp(1.5*x))**2 * ((0.5*(x-10)**2 + 50) - 10 * x**2) 

# x = np.linspace(-5,20,100)
# plt.plot(x,f(x), 'k')
# plt.show()

def gd(x, lr):
    curr = x
    for _ in range(10000):
        prev = curr
        curr = step(prev, lr)
    return curr

def step(x, lr):
    return x - lr * fprime(x)


def find_minima(lr):
    x_converge = []
    start = np.random.uniform(-5, 20, 10)
    for x in start:
        x_converge = x_converge + [gd(x, lr)]
    return x_converge

np.random.seed(9876)
print(find_minima(0.01))

np.random.seed(5432)
print(find_minima(0.3))

np.random.seed(10)
print(find_minima(4))