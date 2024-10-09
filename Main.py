from writers.ParamsWriter import ParamsWriter
from readers.ParamsReader import ParamsReader
from BackwardPass import BackwardPass
from ForwardPass import ForwardPass
from activations.CrossEntropy import CrossEntropy
from optimizations.BatchNorm import BatchNormPerceptron as BatchNormP
import cupy as cp
import numpy as np

if __name__ == "__main__":
    neurons = [2, 10, 10, 10]
    perc_params_file = "otherFiles/perceptron_params.json"
    batch_params_file = "otherFiles/batchnorm_params.json"
    batch_properties_file = "otherFiles/batchnorm_properties.json"
    save_perc_params_file = "otherFiles/last_perceptron_params.json"
    save_batch_params_file = "otherFiles/last_batchnorm_params.json"

    ParamsWriter.writeRandomPerceptronParams(neurons, perc_params_file, save_perc_params_file)
    ParamsWriter.initBatchNormParams(neurons, batch_params_file, save_batch_params_file)
    ParamsWriter.initBatchNormProperties(neurons, batch_properties_file)

    BatchNormP.installFromFile(batch_params_file, batch_properties_file, cp)

    params = ParamsReader.readParams(perc_params_file)

    input_vals = [[12, 9], [15, 15], [21, 24], [52, 0], [60, 5]]
    right_classes = [1, 2, 3, 4, 5]
    summa, activated, norm_summa, scaled_summa, inverse_deviation = ForwardPass.forwardTrainPerceptron(input_vals,
                                                                                                       params, cp)
    do_params = params.copy()
    modified_params = BackwardPass.backwardPerceptron(summa, activated, norm_summa, scaled_summa, inverse_deviation,
                                                   params, right_classes, cp, 0.01)
    gamma = [el.tolist() for el in BatchNormP.gamma]
    beta = [el.tolist() for el in BatchNormP.beta]
    modified_params[0] = [el.tolist() for el in modified_params[0]]
    modified_params[1] = [el.tolist() for el in modified_params[1]]
    ParamsWriter.writeBatchNormParams([gamma, beta], batch_params_file, save_batch_params_file)
    ParamsWriter.writePerceptronParams(modified_params, perc_params_file, save_perc_params_file)
    average = [el.tolist() for el in BatchNormP.getAverage()]
    dispersion = [el.tolist() for el in BatchNormP.getDispersion()]
    ParamsWriter.writeBatchNormProperties([average, dispersion], batch_properties_file)

