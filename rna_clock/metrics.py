from dataclasses import dataclass
from typing import *


@dataclass(frozen=True)
class BasicMetrics:
    MAE: float
    MSE: float
    huber: float

    @staticmethod
    def from_dict(dic: Dict[str, float]):
        return BasicMetrics(dic["l1"], dic["l2"], dic["huber"])

    @staticmethod
    def from_dict(dict: Dict[Dict], row: int):
        return BasicMetrics(dict["l1"][row], dict["l2"][row], dict["huber"][row])

    @staticmethod
    def parse_eval(evals_result: Dict):
        dict = list(evals_result.values())[0]
        l = len(dict["l1"])
        return [BasicMetrics.from_dict(dict, i) for i in range(0, l)]