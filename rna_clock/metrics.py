from dataclasses import dataclass
from typing import *


class BasicMetrics:
    MAE: float
    MSE: float
    huber: float

    def __init__(self, MAE: float, MSE: float, huber: float):
        self.MAE = MAE
        self.MSE = MSE
        self.huber = huber

    @staticmethod
    def from_dict(dic: Dict[str, float]):
        return BasicMetrics(dic["l1"], dic["l2"], dic["huber"])

    @staticmethod
    def from_dict(dict: Dict[str, Dict], row: int):
        return BasicMetrics(dict["l1"][row], dict["l2"][row], dict["huber"][row])

    @staticmethod
    def parse_eval(evals_result: Dict):
        dic = list(evals_result.values())[0]
        return [BasicMetrics.from_dict(dic, i) for i in range(0, len(dic["l1"]))]
