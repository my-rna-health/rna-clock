import numpy
import polars as pl
from dataclasses import dataclass
from typing import *
from functional import seq


def to_XY(expressions: pl.DataFrame, to_predict: str = "medium_age") -> tuple[numpy.ndarray, numpy.ndarray]:
    gene_columns = seq(expressions.columns).filter(lambda s: "ENSG" in s).to_list()
    return expressions.select(gene_columns).to_numpy(), expressions.select(to_predict).to_numpy()


def with_folds(df: pl.DataFrame, folds: int):
    count = pl.lit(df.shape[0])
    per_fold = count / folds
    return df.with_row_count("num").with_column( (pl.col("num") / count).alias("acc")) \
        .with_column((pl.col("num") / per_fold + 1).cast(pl.Int32).alias("fold")).drop("num")


def with_train_test(df: pl.DataFrame, train: float = 0.8):
    count = pl.lit(df.shape[0])
    acc = (pl.col("num") / count).alias("acc")
    fold = pl.when(acc < train).then("train").otherwise("test").alias("fold")
    return df.with_row_count("num").with_column(fold)


def with_train_test_split(df: pl.DataFrame, train: float = 0.8, as_dict: bool = False)-> list[pl.DataFrame] | dict[Any, pl.DataFrame]:
    return with_train_test(df, train).drop("num").partition_by("fold", as_dict=as_dict)


def with_train_test_eval(df: pl.DataFrame, train: float = 0.8, test: float = 0.1):
    count = pl.lit(df.shape[0])
    acc = (pl.col("num") / count).alias("acc")
    fold = pl.when(acc < train).then("train")\
        .otherwise(pl.when(acc < train + test).then("test").otherwise("eval")).alias("fold")
    return df.with_row_count("num").with_column(fold)


def with_train_test_eval_split(df: pl.DataFrame, train: float = 0.8, test: float = 0.1, as_dict: bool = False) -> list[pl.DataFrame] | dict[Any, pl.DataFrame]:
    return with_train_test_eval(df, train, test).drop("num").partition_by("fold", as_dict=as_dict)

