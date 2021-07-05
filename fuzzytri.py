import numpy as np


class FuzzyPy():

    def trimf(self, x, a, b, c):
        X1 = (x - a) / (b - a)
        X2 = (c - x) / (c - b)
        X3 = np.minimum(X1, X2)
        X4 = np.zeros(x.size)
        y = np.maximum(X3, X4)
        return y

    def trapmf(self, x, a, b, c, d):
        X1 = (x - a) / (b - a)
        X2 = np.ones(x.size)
        X3 = (d - x) / (d - c)
        X4 = np.minimum(np.minimum(X1, X2), X3)
        X5 = np.zeros(x.size)
        y = np.maximum(X4, X5)
        return y

    def gaussmf(self, x, c, v):
        """Compute Gaussian Membership function. """
        y = [np.exp(-np.power((i - c), 2) / (2 * v ** 2.0)) for i in x]
        return y

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x)/np.sum(np.exp(x), axis=0)


class Triangle(FuzzyPy):
    def __init__(self, x, low, middle, high):
        self.low = self.trimf(x, low[0], low[1], low[2])
        self.middle = self.trimf(x, middle[0], middle[1], middle[2])
        self.high = self.trimf(x, high[0], high[1], high[2])


class Triangle5(FuzzyPy):
    def __init__(self, x, low2, low1, middle, high1, high2):
        self.low2 = self.trimf(x, low2[0], low2[1], low2[2])
        self.low1 = self.trimf(x, low1[0], low1[1], low1[2])
        self.middle = self.trimf(x, middle[0], middle[1], middle[2])
        self.high1 = self.trimf(x, high1[0], high1[1], high1[2])
        self.high2 = self.trimf(x, high2[0], high2[1], high2[2])


def intersect(A, B):
     """Intersection"""
    return np.minimum(A,B)


def union(A, B):
    """Union"""
    return np.maximum(A, B)

def complement(A):
    """complement"""
    return 1-A


def add(A,B):
    """Adds two fuzzy membership functions/sets"""
    return np.minimum(A+B,1)
