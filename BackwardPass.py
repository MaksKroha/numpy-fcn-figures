from activations.NeuronsActivation import NeuronsActivation as Activation
from activations.CrossEntropy import CrossEntropy
from optimizations.BatchNorm import BatchNormPerceptron as BatchNormP


class BackwardPass:
    # Backward pass is based on back propagation method
    @staticmethod
    def backwardPerceptron(summa, activated: list, norm_summa: list, scaled_summa: list,
                           inverse_deviation: list, params: list, right_classes: list,
                           realization, step, epochs):
        batches_num = summa.shape[0]
        loss_func = CrossEntropy(summa, right_classes, realization)
        bias_grads = loss_func.getDerivative()
        weights_grads = activated.pop() * bias_grads

        params[0][-1] -= step * weights_grads
        params[1][-1] -= step * bias_grads

        for layer in range(len(params[1]) - 2, -1, -1):
            # TODO: need to divide by number of batches. but should check it
            # TODO: need to do transposition where it needed
            beta_grads = Activation.derivative(scaled_summa.pop(), realization)\
                         * bias_grads * params[0][layer + 1]
            gamma_grads = norm_summa.pop() * beta_grads
            bias_grads = inverse_deviation.pop() * BatchNormP.gamma[layer] * gamma_grads
            weights_grads = bias_grads.T * activated.pop()

            BatchNormP.beta[layer] -= step * beta_grads
            BatchNormP.gamma[layer] -= step * gamma_grads
            params[0][layer] -= step * weights_grads
            params[1][layer] -= step * bias_grads

