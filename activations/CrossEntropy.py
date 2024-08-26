import cupy as cp
import numpy as np
import math


# axis=0 : применить вычисление «по столбцам»
# axis=1 : применить вычисление «построчно»
# CrossEntropy for mini-batches gradient descent
class CrossEntropy:
    def __init__(self, logits, right_class, realization):
        # logits shape must be n x numOfBatches
        self.probabilities = []
        self.__softmax__(logits, realization)
        self.derivative = self.probabilities
        # Need to review
        self.derivative[right_class] -= 1

    def getDerivative(self):
        return self.derivative

    def __softmax__(self, logits, realization):
        denominator = 0
        # searching max logits of each column
        max_logits = realization.max(logits, axis=0)
        # from each column subtract max logit of each column
        # and finding "e" in this degree
        exp_logits = realization.exp(logits - max_logits)
        # finding a sum of each column - it is a dominators
        denominators = realization.sum(exp_logits, axis=0)
        self.probabilities = exp_logits / denominators