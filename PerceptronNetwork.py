import numpy as np
import cupy as cp

from activations.NeuronsActivation import NeuronsActivation as Activation


class PerceptronNetwork:
    @staticmethod
    def forward(input_vals, params, realization):
        # input_vals there is a mini-batch with input data vectors
        # params there are all weights and biases
        # params[0] - weights, params[1] - biases
        layers = len(params[0])
        params[0] = realization.array(params[0])
        params[1] = realization.array(params[1])
        input_vals = realization.array(input_vals)
        activated = [input_vals]
        summa = []

        for layer in range(layers):
            # NEED TO REVIEW
            summa.append(params[0][layer] * activated[-1] + params[1][layer])
            activated.append(Activation.activate(summa[-1], realization))

        return summa, activated[0:-1]