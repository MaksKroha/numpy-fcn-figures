import numpy as np
from numpy.typing import NDArray as nparray
from readers.ParamsReader import ParamsReader


class BatchNormPerceptron:
    average = None
    dispersion = None
    gamma = None
    beta = None
    alpha = 0.99

    @classmethod
    def installFromFile(cls, batchnorm_params_file, batchnorm_properties_file, realization):
        # properties[0] - average
        # properties[1] - dispersion
        params = ParamsReader.readParams(batchnorm_params_file)
        properties = ParamsReader.readProperties(batchnorm_properties_file)
        for i in range(len(properties[0])):
            params[0][i] = realization.array(params[0][i])
            params[1][i] = realization.array(params[1][i])
            properties[0][i] = realization.array(properties[0][i])
            properties[1][i] = realization.array(properties[1][i])
        cls.average = properties[0]
        cls.dispersion = properties[1]
        cls.gamma = params[0]
        cls.beta = params[1]

    @classmethod
    def getDispersion(cls):
        return cls.dispersion

    @classmethod
    def getAverage(cls):
        return cls.average

    @classmethod
    def updateDispersion(cls, dispersion, layer):
        cls.dispersion[layer] = cls.alpha * cls.dispersion[layer] + (1 - cls.alpha) * dispersion

    @classmethod
    def updateAverage(cls, average, layer):
        cls.average[layer] = cls.alpha * cls.average[layer] + (1 - cls.alpha) * average
