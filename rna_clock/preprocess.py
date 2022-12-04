from typing import Mapping, Union

import polars as pl
from functional import seq
from polars import DataType

ensembl_gene_col = pl.col("^ENSG[a-zA-Z0-9]+$")

def extract_ensembl_columns(expressions: pl.DataFrame) -> list[str]:
    gene_columns = seq(expressions.columns).filter(lambda s: "ENSG" in s).to_list()
    return gene_columns


def cols_of(cols: list[str], dt: DataType = pl.Float32) -> Mapping[str, type[DataType]]:
    return {c: dt for c in cols}


def col_sum(exclude: Union[str, list[str]], final_name: str = "sum"):
    return pl.fold(acc=pl.lit(0.0), f=lambda acc, x: acc + x, exprs=pl.all().exclude(exclude)).alias(final_name)


def binary_with_tpm_row_average(df: pl.DataFrame, exclude: list[str], row_sum: float = 1e6):
    avg = row_sum / (len(df.columns) - len(exclude))
    other_cols = seq(exclude).map(lambda s: pl.col(s)).to_list()
    to_select = other_cols + [pl.exclude(exclude) > avg]
    return df.select(to_select)


def with_row_average(df: pl.DataFrame,
                     exclude: Union[str, list[str]],
                     final_name: str = "average", size: int = None):
    length = len(df.columns) - 1 if size is None else size
    return df.with_column(col_sum(exclude, final_name) / length)


def sum_rows(df: pl.DataFrame, name: Union[str, list[str]]):
    df.select(
        pl.fold(acc=pl.lit(0.0), f=lambda acc, x: acc + x, exprs=pl.all().exclude(name)).alias("sum"),
    )
    return df.select([pl.col(name), pl.all().exclude(name).sum().alias("row_sum")])


def binarization(df: pl.DataFrame, name: str):
    col = pl.col(name)
    col / col.sum()
    return df.select(col)