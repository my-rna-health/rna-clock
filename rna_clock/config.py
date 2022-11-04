import polars as pl
from pathlib import Path
from pycomfort.files import *
from pycomfort import files
import pyarrow
import pandas as pd
from functional import seq
from typing import *
from rna_clock.splits import *
from rna_clock import *
from rna_clock.trees import *
import sys


class Locations:
    base: Path
    data: Path
    gtex: Path

    gtex_input: Path
    gtex_interim: Path
    gtex_output: Path

    cattle: Path
    cattle_input: Path
    cattle_output: Path

    ensembl: Path
    coding_human_genes: Path

    def __init__(self, base: Path):
        self.base = base.absolute().resolve()
        self.data = self.base / "data"
        self.gtex = self.data / "gtex"
        self.gtex.mkdir(parents=True, exist_ok=True)

        self.gtex_input = self.gtex / "input"
        self.gtex_input.mkdir(parents=True, exist_ok=True)

        self.gtex_interim = self.gtex / "interim"
        self.gtex_interim.mkdir(parents=True, exist_ok=True)
        self.cattle = self.data / "cattle"
        self.cattle_input = self.cattle / "input"
        self.cattle_interim = self.cattle / "interim"
        self.cattle_output = self.cattle / "output"

        self.gtex_output = self.gtex / "output"
        self.ensembl = self.data / "ensembl"
        self.coding_human_genes = self.ensembl / "coding_human_genes.tsv"


from enum import Enum
from typing import Dict

gtex_parameters: Dict = dict(
    {"objective": "regression",
     'boosting_type': 'gbdt',
     'lambda_l1': 2.649670285109348,
     'lambda_l2': 3.651743005278647,
     'max_leaves': 21,
     'max_depth': 3,
     'feature_fraction': 0.7381836300988616,
     'bagging_fraction': 0.5287709904685758,
     'learning_rate': 0.054438364299744225,
     'min_data_in_leaf': 7,
     'drop_rate': 0.13171689004108006,
     'metric': ['mae', 'mse', 'huber'],
     }
)