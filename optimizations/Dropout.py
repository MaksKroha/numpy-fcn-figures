import numpy as np
import numpy.random


class Dropout:
    # dropout is performed on CPU
    # due to not big calculations
    @staticmethod
    def getDropoutMatrix(matrix: np.array, prob):
        # prob - it is the probability with
        # which neurons will remain activated

        # "*" deploys a tuple
        binary_mask = numpy.random.rand(*matrix.shape) < prob
        return matrix * binary_mask / prob, binary_mask