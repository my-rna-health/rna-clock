from typing import *

import lightgbm as lgb
from lightgbm import Booster
from metrics import *


def train_light_gbm(self, X_train, X_test, y_train, y_test,
                     parameters: Dict, categorical=None,
                     num_boost_round: int = 250, seed: int = None) -> [Booster, list[BasicMetrics]]:
    cat = categorical if (categorical is not None) and len(categorical) > 0 else "auto"
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    evals_result = {}

    stopping_callback = lgb.early_stopping(self.early_stopping_rounds)
    if seed is not None:
        parameters["seed"] = seed
    gbm = lgb.train(parameters,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_eval],
                    evals_result=evals_result,
                    verbose_eval=num_boost_round,
                    callbacks=[stopping_callback]
                    )
    return gbm, BasicMetrics.parse_eval(evals_result)

def validate_lightgbm()