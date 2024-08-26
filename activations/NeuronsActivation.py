class NeuronsActivation:
    # all those methods are for hidden layers and input
    @staticmethod
    def activate(mat, realization):
        # Leaky ReLU
        # "mat" type must be like realization
        return realization.where(mat > 0, mat, 0.01 * mat)

    @staticmethod
    def derivative(mat, realization):
        return realization.where(mat > 0, 1, 0.01)
