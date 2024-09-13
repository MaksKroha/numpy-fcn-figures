import numpy as np
from numpy.typing import NDArray as nparray


class BatchNormPerceptron:
    average = None
    dispersion = None
    gamma = None
    beta = None
    alpha = 0.9

    @classmethod
    def initialize(cls, biases, realization):
        ones = [realization.ones_like(vector) for vector in biases]
        zeros = [realization.zeros_like(vector) for vector in biases]
        cls.average, cls.beta = zeros
        cls.dispersion, cls.gamma = ones

    @classmethod
    def getDispersion(cls):
        return cls.dispersion

    @classmethod
    def getAverage(cls):
        return cls.average

    @classmethod
    def updateDispersion(cls, dispersion):
        cls.dispersion = cls.alpha * cls.dispersion + (1 - cls.alpha) * dispersion

    @classmethod
    def updateAverage(cls, average):
        cls.average = cls.alpha * cls.average + (1 - cls.alpha) * average
