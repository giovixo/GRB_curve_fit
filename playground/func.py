
import numpy as np
import matplotlib.pyplot as plt

class Parabola:
    def __init__(self, a=1, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c
    def f(self, x):
        return self.a * x**2 + self.b * x + self.c
    def show(self):
        print "a = {}, b = {}, c={}".format(self.a, self.b, self.c)
    def plot(self):
        x = np.linspace(-10,10,100)
        plt.plot(x, self.f(x))
        plt.show()
