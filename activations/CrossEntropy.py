import cupy as cp
import numpy as np


# axis=0 : применить вычисление «по столбцам»
# axis=1 : применить вычисление «построчно»
# CrossEntropy for mini-batches gradient descent
class CrossEntropy:
    def __init__(self, logits, right_classes: list, realization):
        # logits shape must be n x numOfBatches
        self.probabilities = []
        self.__softmax__(logits, realization)
        self.derivative = self.probabilities.copy()
        # Need to review
        for line in range(len(self.derivative)):
            self.derivative[line][right_classes[line]] -= 1

    def getDerivative(self):
        return self.derivative

    def getProbabilities(self):
        return self.probabilities

    def __softmax__(self, logits, realization):
        # searching max logits of each column
        max_logits = realization.max(logits, axis=1, keepdims=True)
        # from each column subtract max logit of each column
        # and finding "e" in this degree
        exp_logits = realization.exp(logits - max_logits)
        # finding a sum of each column - it is a dominators
        denominators = realization.sum(exp_logits, axis=1, keepdims=True)
        self.probabilities = exp_logits / denominators
