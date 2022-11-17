from dataclasses import dataclass
from pathlib import Path
from typing import *

from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class BasicMetrics:
    MAE: float
    MSE: float
    huber: float

    @staticmethod
    def to_json_array(array: List['BasicMetrics']):
        return BasicMetrics.schema().dumps(array, many=True)

    @staticmethod
    def write_json_array(array: List['BasicMetrics'], where: Path):
        arr = BasicMetrics.to_json_array(array)
        where.touch(exist_ok=True)
        where.write_text(arr, encoding="utf-8")
        return where

    def write_json(self, where: Path):
        where.touch(exist_ok=True)
        txt = self.to_json()
        where.write_text(txt, encoding="utf-8")
        return where

    @staticmethod
    def from_light_dict(dic: Dict[str, float]):
        return BasicMetrics(dic["l1"], dic["l2"], dic["huber"])

    @staticmethod
    def from_light_dict_row(dic: Dict[str, Dict], row: int):
        return BasicMetrics(dic["l1"][row], dic["l2"][row], dic["huber"][row])

    @staticmethod
    def parse_eval(evals_result: Dict):
        length = len(list(evals_result.values())[0])
        return [BasicMetrics.from_light_dict_row(evals_result, i) for i in range(0, length)]
