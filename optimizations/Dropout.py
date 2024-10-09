import numpy as np
import numpy.random


class Dropout:
    # dropout is performed on CPU
    # due to not big calculations
    @staticmethod
    def getDropoutMatrix(matrix, remain_probability, realization):
        # prob - it is the probability with
        # which neurons will remain activated

        # "*" deploys a tuple
        binary_mask = realization.random.rand(*matrix.shape) < remain_probability
        matrix = matrix * binary_mask / remain_probability
        return matrix