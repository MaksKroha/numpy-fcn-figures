import numpy as np
import json as js


class ParamsWriter:
    @staticmethod
    def writeRandomPerceptronParams(layers, neurons: list):
        params = [[], []]
        for i in range(1, len(neurons)):
            params[0].append(np.random.uniform(-0.05, 0.05, size=(neurons[i], neurons[i - 1])))
            params[1].append(np.random.uniform(-0.05, 0.05, size=(1, neurons[i])))
        to_json = {"params": params}
        # TODO: Autosaver last data
        with open("../otherFiles/perceptron_params.json", 'w') as file:
            js.dump(to_json, file)

    @staticmethod
    def writePerceptronParams():
        # TODO

    @staticmethod
    def resetBatchNormParams():
        # TODO

    @staticmethod
    def writeBatchNormParams():
        # TODO