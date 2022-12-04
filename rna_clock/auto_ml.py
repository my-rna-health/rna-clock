from flaml import AutoML
from rna_clock.splits import *
import polars as pl

def get_automl(expressions: pl.DataFrame, time_budget: int =None):
    [train, test] = with_train_test_split(expressions)
    train_X, train_Y = to_XY(train)
    automl = AutoML()
    automl.fit(train_X, train_Y, task="regression", time_budget=time_budget)
    test_X, test_Y = to_XY(test)
    score = automl.score(test_X, test_Y)
    return automl, score