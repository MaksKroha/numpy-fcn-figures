import numpy as np


class DatasetReader:
    @staticmethod
    def getLabelsAndFiguresList(path, realization):
        symbols_mat = realization.loadtxt(path, delimiter=',', skiprows=0).astype(int)
        return symbols_mat
