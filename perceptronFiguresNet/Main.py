from writers.ParamsWriter import ParamsWriter
from readers.DatasetReader import DatasetReader
from readers.ParamsReader import ParamsReader
from BackwardPass import BackwardPass
from ForwardPass import ForwardPass
from activations.CrossEntropy import CrossEntropy
from optimizations.BatchNorm import BatchNormPerceptron as BatchNormP
import cupy as cp
import numpy as np

if __name__ == "__main__":
    # Just static paths to files
    perc_params_file = "../otherFiles/perceptron_params.json"
    batch_params_file = "../otherFiles/batchnorm_params.json"
    batch_properties_file = "../otherFiles/batchnorm_properties.json"
    save_perc_params_file = "../otherFiles/last_perceptron_params.json"
    save_batch_params_file = "../otherFiles/last_batchnorm_params.json"
    mnist_train_file = "../otherFiles/mnist/mnist_train.csv"
    mnist_test_file = "../otherFiles/mnist/mnist_test.csv"
    realization = np
    isCupyRealization = False
    checker = False
    if checker:
        neurons = [784, 784, 784, 10]
        step = 0.001
        epochs = 200
        batches_num = 1028
        train_element_num = 60_000

        dataset = DatasetReader.getLabelsAndFiguresList(mnist_train_file, realization)
        # Initializing and Writing Random parameters
        # ParamsWriter.writeRandomPerceptronParams(neurons, perc_params_file, save_perc_params_file)
        # ParamsWriter.initBatchNormParams(neurons, batch_params_file, save_batch_params_file)
        # ParamsWriter.initBatchNormProperties(neurons, batch_properties_file)

        # installing all Parameters into a static class Batch Norm
        BatchNormP.installFromFile(batch_params_file, batch_properties_file, realization)

        # Reading all needed data from files
        params = ParamsReader.readParams(perc_params_file)

        if isCupyRealization:
            dataset = cp.asarray(dataset)

        for _ in range(epochs):
            print(_)
            realization.random.shuffle(dataset)
            for i in range(0, train_element_num, batches_num):
                mini_batch = dataset[i: i + batches_num]
                labels_list = mini_batch[:, 0]
                figures_list = mini_batch[:, 1:] / 255
                r1, r2, r3, r4, r5 = ForwardPass.forwardTrainPerceptron(figures_list, params, realization)
                params = BackwardPass.backwardPerceptron(r1, r2, r3, r4, r5, params, labels_list.tolist(), realization, step)

        gamma = [el.tolist() for el in BatchNormP.gamma]
        beta = [el.tolist() for el in BatchNormP.beta]
        params[0] = [el.tolist() for el in params[0]]
        params[1] = [el.tolist() for el in params[1]]
        ParamsWriter.writeBatchNormParams([gamma, beta], batch_params_file, save_batch_params_file)
        ParamsWriter.writePerceptronParams(params, perc_params_file, save_perc_params_file)
        average = [el.tolist() for el in BatchNormP.getAverage()]
        dispersion = [el.tolist() for el in BatchNormP.getDispersion()]
        ParamsWriter.writeBatchNormProperties([average, dispersion], batch_properties_file)

    # Testing
    dataset_test = DatasetReader.getLabelsAndFiguresList(mnist_test_file, np)
    for i in range(3000, 3020):
        BatchNormP.installFromFile(batch_params_file, batch_properties_file, np)
        params = ParamsReader.readParams(perc_params_file)
        summa = ForwardPass.forwardTestPerceptron(dataset_test[i, 1:] / 255, params)
        cross = CrossEntropy(summa, [dataset_test[i, 0]], np)
        print(f"expected - {dataset_test[i, 0]},  reality - {cross.getProbabilities()}")
