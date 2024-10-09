from activations.NeuronsActivation import NeuronsActivation as Activation
from optimizations.BatchNorm import BatchNormPerceptron as BatchNormP
from optimizations.Dropout import Dropout
import cupy as cp
import numpy as np


class ForwardPass:
    isCupyMat = None

    @staticmethod
    def forwardTrainPerceptron(input_vals, params, realization, dropout_probability=0.7):
        # input_vals there is a mini-batch with input data vectors
        # params there are all weights and biases
        # params[0] - weights, params[1] - biases
        isCupyMat = isinstance(realization.ones((1, 1)), cp.ndarray)
        layers = len(params[1])
        batches_num = len(input_vals)
        params[0] = [realization.array(el) for el in params[0]]
        params[1] = [realization.array(el) for el in params[1]]
        input_vals = realization.array(input_vals)
        activated = [input_vals]
        summa = None

        # norm_summa it is a summa, which is
        # normalized by batch-normalization
        norm_summa = []
        scaled_summa = []
        inverse_deviation = []
        for layer in range(layers - 1):
            summa = realization.dot(activated[-1], params[0][layer]) + params[1][layer]
            average = realization.mean(summa, axis=0)
            # print(f"batches num - 1 = {realization.sum((summa - average) ** 2, axis=0)}")
            dispersion = realization.sum((summa - average) ** 2, axis=0) / batches_num
            standard_deviation = realization.sqrt(dispersion + 10 ** -8)

            norm_summa.append((summa - average) / standard_deviation)
            scaled_summa.append(norm_summa[-1] * BatchNormP.gamma[layer] + BatchNormP.beta[layer])
            activated_val = Activation.activate(scaled_summa[-1], realization)
            activated.append(Dropout.getDropoutMatrix(activated_val, dropout_probability, realization))

            # refresh the global average
            # and global dispersion with exponential average law
            BatchNormP.updateAverage(average, layer)
            BatchNormP.updateDispersion(dispersion, layer)

            # needed for gradient descent
            inverse_deviation.append(1 / standard_deviation)

        summa = realization.dot(activated[-1], params[0][-1]) + params[1][-1]
        return summa, activated, norm_summa, scaled_summa, inverse_deviation

    @staticmethod
    def forwardTestPerceptron(input_vals, params):
        input_vals = np.array(input_vals)
        params[0] = [np.array(el) for el in params[0]]
        params[1] = [np.array(el) for el in params[1]]
        layers = len(params[1])
        summa = None
        activated = [input_vals]
        for layer in range(layers):

            summa = np.dot(activated[-1].T, params[0][layer]) + params[1][layer]
            if layer < layers - 1:
                norm_summa = (summa - BatchNormP.getAverage()[layer]) \
                        / np.sqrt(BatchNormP.getDispersion()[layer] + 10 ** -8)
                scaled_summa = norm_summa * BatchNormP.gamma[layer] + BatchNormP.beta[layer]
                activated = Activation.activate(scaled_summa, np)
        return summa
