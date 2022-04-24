import numpy as np

class Parameter:
    def __init__(self, name, shape, weight_sd, c=0):
        self.name = name
        self.value = np.random.rand(shape[0], shape[1]) * weight_sd + c
        self.diff = np.zeros_like(self.value)
        self.m = np.zeros_like(self.value)