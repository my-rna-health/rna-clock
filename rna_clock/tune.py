import json

from rna_clock.config import Locations
from rna_clock.splits import *
from rna_clock.trees import *
import optuna


def tune_lightgbm_model(X_train, X_test, y_train, y_test,
         params: Dict = None, categorical=None,
         num_boost_round: int = 250, seed: int = 0, early_stopping_rounds = 5, validation_name: str = "validation"):
    time_budget_seconds = 3600
    if params is None:  params = config.gtex_parameters
    cat = categorical if (categorical is not None) and len(categorical) > 0 else "auto"
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    eval_result: Evaluation = dict()
    stopping_callback = lgb.early_stopping(early_stopping_rounds)
    record_evaluation_callback = lgb.record_evaluation(eval_result)
    if seed is not None:
        params["seed"] = seed
    tuner = optuna.integration.lightgbm.LightGBMTuner(
        params, lgb_train,
        valid_sets = [lgb_eval],
        num_boost_round=num_boost_round,
        verbose_eval=num_boost_round,
        time_budget= time_budget_seconds,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[stopping_callback, record_evaluation_callback]
    )
    tuner.tune_bagging()
    tuner.tune_feature_fraction()
    tuner.tune_min_data_in_leaf()
    tuner.tune_feature_fraction_stage2()
    tuner.run()
    best_params: Dict[str, Any] = tuner.best_params
    best_value = tuner.best_score
    return best_params, best_value

def split_and_tune(expressions: pl.DataFrame, for_selection: list[str]) -> [Booster, list[BasicMetrics]]:
    df = expressions.select(for_selection)
    [train, dev, test] = with_train_test_eval_split(df)
    print(f"train [{train.shape}], dev [{dev.shape}], test [{test.shape}]")
    (train_X, train_Y) = to_XY(train)
    (dev_X, dev_Y) = to_XY(train)
    return tune_lightgbm_model(train_X, dev_X, train_Y, dev_Y, validation_name="development")


def tune_group(expressions: pl.DataFrame, for_selection: list[str], group: str, locations: Locations):
    print("writing best tuned params")
    best_params, best_value = split_and_tune(expressions, for_selection)
    print("BEST PARAMS", best_params)
    print("BEST VALUE =", best_value)
    group_output = locations.gtex_output / group
    group_output.mkdir(exist_ok=True)
    par_output = (group_output / "params.json")
    par_output.touch(exist_ok=True)
    with par_output.open("w") as outfile:
        json.dump(best_params, outfile)
    print("TUNED!")

def tune_gtex(locations: Locations):
    print("tune_gtex, tuning best predictions for medium age")
    coding_genes = pl.read_csv(locations.coding_human_genes, sep="\t")
    coding_gene_ids = coding_genes.select(pl.col("Gene stable ID")).to_dict(False)['Gene stable ID']
    expressions = pl.read_parquet(locations.gtex_interim / "expressions_extended.parquet")
    gene_columns = seq(expressions.columns).filter(lambda s: "ENSG" in s)
    coding_gene_columns = sorted(set(coding_gene_ids) & set(gene_columns), key = coding_gene_ids.index)
    to_predict: str = "medium_age"
    for_selection = coding_gene_columns + [to_predict]
    tune_group(expressions, for_selection, "all", locations)