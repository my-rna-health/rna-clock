import lightgbm as lgb
import numpy
import polars as pl
from functional import seq
from lightgbm import Booster

from rna_clock import config
from rna_clock.metrics import *

Evaluation = Dict[str, Dict[str, List[Any]]]

def to_XY(expressions: pl.DataFrame, to_predict: str = "medium_age") -> tuple[numpy.ndarray, numpy.ndarray]:
    gene_columns = seq(expressions.columns).filter(lambda s: "ENSG" in s).to_list()
    return expressions.select(gene_columns).to_numpy(), expressions.select(to_predict).to_numpy()


def train_lightgbm_model(X_train, X_test, y_train, y_test,
                         params: Dict = None, categorical=None,
                     num_boost_round: int = 250, seed: int = 0, early_stopping_rounds = 5, validation_name: str = "validation") -> [Booster, Evaluation]:
    '''
    trains a regression model
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param categorical:
    :param parameters:
    :return:
    '''
    if params is None:
        params = config.gtex_parameters
    cat = categorical if (categorical is not None) and len(categorical) > 0 else "auto"
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    eval_result: Evaluation = dict()
    stopping_callback = lgb.early_stopping(early_stopping_rounds)
    record_evaluation_callback = lgb.record_evaluation(eval_result)
    if seed is not None:
        params["seed"] = seed
    booster = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_eval],
                    valid_names=[validation_name],
                    callbacks=[stopping_callback, record_evaluation_callback]
                    )
    return booster, BasicMetrics.parse_eval(eval_result[validation_name])