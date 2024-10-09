import json as js
import numpy as np


class ParamsReader:
    @staticmethod
    def readParams(fileName) -> list:
        try:
            with open(fileName) as file:
                data = js.load(file)
                return data['params']
        except FileNotFoundError:
            print(f"file {fileName} not found")

    @staticmethod
    def readProperties(fileName):
        try:
            with open(fileName) as file:
                data = js.load(file)
                return data['properties']
        except FileNotFoundError:
            print(f"file {fileName} not found")