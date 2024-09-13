from activations.NeuronsActivation import NeuronsActivation as Activation
from optimizations.BatchNorm import BatchNormPerceptron as BatchNormP
from numpy.typing import NDArray as nparray
import cupy as cp
import numpy as np


class ForwardPass:
    isCupyMat = None

    @staticmethod
    def forwardTrainPerceptron(input_vals, params, realization):
        # input_vals there is a mini-batch with input data vectors
        # params there are all weights and biases
        # params[0] - weights, params[1] - biases
        isCupyMat = isinstance(realization.ones((1, 1)), cp.ndarray)
        layers = len(params[0])
        batches_num = len(input_vals)
        params[0] = realization.array(params[0])
        params[1] = realization.array(params[1])
        input_vals = realization.array(input_vals)
        activated = []
        summa = input_vals
        # norm_summa it is a summa, which is
        # normalized by batch-normalization
        norm_summa = []
        scaled_summa = []
        inverse_deviation = []
        BatchNormP.initialize(params[1], realization)

        for layer in range(layers - 1):
            summa = params[0][layer] * activated[-1] + params[1][layer]

            average = realization.mean(summa, axis=0)
            dispersion = realization.sum((summa - average) ** 2, axis=0) / batches_num - 1
            standard_deviation = realization.sqrt(dispersion + 10**-8)

            norm_summa.append((summa - average) / standard_deviation)
            scaled_summa.append(norm_summa * BatchNormP.gamma[layer] + BatchNormP.beta[layer])
            activated.append(Activation.activate(norm_summa[-1], realization))
            # refresh the global average
            # and global dispersion with exponential average law
            BatchNormP.updateAverage(average)
            BatchNormP.updateDispersion(dispersion)



            # needed for gradient descent
            inverse_deviation.append(1 / standard_deviation)
        summa = params[0][-1] * activated[-1] + params[1][-1]
        return summa, activated, norm_summa, scaled_summa, inverse_deviation

    @staticmethod
    def forwardTestPerceptron(input_vals: nparray, params: list[list[nparray]]):
        layers = len(params[1])
        summa = None
        activated = input_vals
        for layer in range(layers):
            summa = params[0][layer] * activated[-1] + params[1][layer]
            if layer < layers - 1:
                summa = (summa - BatchNormP.average) / np.sqrt(BatchNormP.dispersion + 10**-8)
                activated = Activation.activate(summa, np)
        return summa