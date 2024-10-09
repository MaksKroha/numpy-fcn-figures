from activations.NeuronsActivation import NeuronsActivation as Activation
from activations.CrossEntropy import CrossEntropy
from optimizations.BatchNorm import BatchNormPerceptron as BatchNormP


class BackwardPass:
    # Backward pass is based on back propagation method
    @staticmethod
    def backwardPerceptron(summa, activated: list, norm_summa: list, scaled_summa: list,
                           inverse_deviation: list, params: list, right_classes: list,
                           realization, step):
        # backward pass is realized for one mini-batch
        batches_num = summa.shape[0]
        loss_func = CrossEntropy(summa, right_classes, realization)
        bias_grads = loss_func.getDerivative()
        weights_grads = realization.dot(activated.pop().T, bias_grads) / batches_num

        params[0][-1] -= step * weights_grads
        params[1][-1] -= step * realization.mean(bias_grads, axis=0)

        for layer in range(len(params[1]) - 2, -1, -1):
            temp = realization.dot(bias_grads, params[0][layer + 1].T)
            beta_grads = temp * Activation.derivative(scaled_summa.pop(), realization) / batches_num
            gamma_grads = norm_summa.pop() * beta_grads
            bias_grads = inverse_deviation.pop() * BatchNormP.gamma[layer] * beta_grads
            weights_grads = realization.dot(bias_grads.T, activated.pop()) / batches_num
            BatchNormP.beta[layer] -= step * realization.mean(beta_grads, axis=0)
            BatchNormP.gamma[layer] -= step * realization.mean(gamma_grads, axis=0)
            params[0][layer] -= step * weights_grads.T
            params[1][layer] -= step * realization.mean(bias_grads, axis=0)
        return params
