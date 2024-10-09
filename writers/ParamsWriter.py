from readers.ParamsReader import ParamsReader
import numpy as np
import json as js


class ParamsWriter:
    @staticmethod
    def writeRandomPerceptronParams(neurons: list, filePath, file_saver):
        ParamsWriter.transfer(filePath, file_saver)
        try:
            params = [[], []]
            for i in range(0, len(neurons) - 1):
                params[0].append(np.random.uniform(-0.1, 0.1, size=(neurons[i], neurons[i + 1])).tolist())
                params[1].append(np.random.uniform(-0.1, 0.1, size=(1, neurons[i + 1])).tolist())
            to_json = {"params": params}
            with open(filePath, 'w') as file:
                js.dump(to_json, file)
        except FileNotFoundError:
            print("file with perceptron params not found")

    @staticmethod
    def writePerceptronParams(params: list[list[list]], file_path, file_saver):
        ParamsWriter.transfer(file_path, file_saver)
        try:
            to_json = {"params": params}
            with open(file_path, 'w') as file:
                js.dump(to_json, file)
        except FileNotFoundError:
            print("file with perceptron params not found")

    @staticmethod
    def initBatchNormParams(neurons, file_path, file_saver):
        ParamsWriter.transfer(file_path, file_saver)
        try:
            params = [[], []]
            for i in range(1, len(neurons) - 1):
                # params[0] - gamma
                # params[1] - beta
                params[0].append(np.ones((1, neurons[i])).tolist())
                params[1].append(np.zeros((1, neurons[i])).tolist())
            to_json = {"params": params}
            with open(file_path, 'w') as file:
                js.dump(to_json, file)
        except FileNotFoundError:
            print("file with batchnorm params not found")

    @staticmethod
    def writeBatchNormParams(params: list[list], file_path, file_saver):
        # properties[0] - average
        # properties[1] - dispersion
        ParamsWriter.transfer(file_path, file_saver)
        try:
            to_json = {"params": params}
            with open(file_path, 'w') as file:
                js.dump(to_json, file)
        except FileNotFoundError:
            print("file with perceptron params not found")

    @staticmethod
    def initBatchNormProperties(neurons, file_path):
        try:
            properties = [[], []]
            for i in range(1, len(neurons) - 1):
                # properties[0] - average
                # properties[1] - dispersion
                properties[0].append(np.zeros((1, neurons[i])).tolist())
                properties[1].append(np.ones((1, neurons[i])).tolist())
            to_json = {"properties": properties}
            with open(file_path, 'w') as file:
                js.dump(to_json, file)
        except FileNotFoundError:
            print("file with batchnorm properties not found")

    @staticmethod
    def writeBatchNormProperties(properties: list[list], file_path):
        # properties[0] - average
        # properties[1] - dispersion
        try:
            to_json = {"properties": properties}
            with open(file_path, 'w') as file:
                js.dump(to_json, file)
        except FileNotFoundError:
            print("file with batch norm properties not found")

    @staticmethod
    def transfer(source_file, target_file):
        try:
            some_data = ParamsReader.readParams(source_file)
            if some_data:
                to_json = {"params": some_data}
                with open(target_file, 'w') as file:
                    js.dump(to_json, file)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            return