import lightgbm as lgb
import numpy
import polars as pl
from dataclasses import dataclass
from typing import *
from functional import seq

def to_XY(expressions: pl.DataFrame, to_predict: str = "medium_age") -> tuple[numpy.ndarray, numpy.ndarray]:
    gene_columns = seq(expressions.columns).filter(lambda s: "ENSG" in s).to_list()
    return expressions.select(gene_columns).to_numpy(), expressions.select(to_predict).to_numpy()
