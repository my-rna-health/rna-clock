import json

from lightgbm import CVBooster
from optuna import Trial, Study

from rna_clock.config import Locations
from rna_clock.splits import *
from rna_clock.trees import *
import optuna
from optuna import trial
from rna_clock.preprocess import ensembl_gene_col

def tune_lightgbm_model(dataset: lgb.Dataset, n_fold: int = 5, stratified: bool = False,
         params: Dict = None,
         num_boost_round: int = 1000, seed: int = 0, early_stopping_rounds = 50,
                        locations: Locations = None, group: str = "all") -> (Dict[str, Any], float, Booster):
    time_budget_seconds = 12000
    if params is None:
        params = config.gtex_parameters
        params["metric"] = "mae"
    eval_result: Evaluation = dict()
    #stopping_callback = lgb.early_stopping(early_stopping_rounds)
    #record_evaluation_callback = lgb.record_evaluation(eval_result)
    if seed is not None:
        params["seed"] = seed
    storage: optuna.storages.RDBStorage = None
    model_dir: str = None
    if locations is not None:
        to_save = str((locations.gtex_interim / f"{group}_lightgbm.sqlite").absolute())
        print(f'trial will be saved to: {to_save}')
        url = f'sqlite:///' + to_save
        storage = optuna.storages.RDBStorage(
            url=url
            # engine_kwargs={'check_same_thread': False}
        )
        model_dir_path: Path = locations.gtex_interim / f"{group}"
        model_dir_path.mkdir(parents=True, exist_ok=True)
        model_dir = str(model_dir_path)

    study: Study = optuna.create_study(storage=storage, study_name=f"gtex_lightgbm_optimization_for_{group}", load_if_exists=True, direction="minimize")
    tuner = optuna.integration.lightgbm.LightGBMTunerCV(
        model_dir=model_dir,
        params = params,
        train_set = dataset,
        num_boost_round=num_boost_round,
        stratified = stratified,
        study = study,
        nfold = n_fold,
        verbose_eval=num_boost_round,
        time_budget=time_budget_seconds,
        early_stopping_rounds=early_stopping_rounds,
        #callbacks=[stopping_callback, record_evaluation_callback],
        return_cvbooster=True
    )
    #tuner.tune_bagging(20)
    #tuner.tune_feature_fraction(14)
    #tuner.tune_num_leaves(40)
    #tuner.tune_regularization_factors(40)
    #tuner.tune_min_data_in_leaf()
    #tuner.tune_feature_fraction_stage2(12)
    tuner.run()
    best_params: Dict[str, Any] = tuner.best_params
    best_value: float = tuner.best_score
    best_booster = tuner.get_best_booster()

    return best_params, best_value, best_booster


def tune_group(expressions: pl.DataFrame, for_selection: list[pl.col], group: str, locations: Locations):
    [train, test] = with_train_test_split(expressions.select(for_selection), 0.2)
    print(f"train [{train.shape}], test [{test.shape}]")
    (train_X, train_Y) = to_XY(train)
    (test_X, test_Y) = to_XY(train)
    train_dataset = lgb.Dataset(train_X, train_Y)
    test_dataset = lgb.Dataset(test_X, test_Y)
    best_params, best_value, booster = tune_lightgbm_model(train_dataset, group=group)
    print("BEST PARAMS", best_params)
    print("BEST VALUE =", best_value)
    print("=======Evaluations:===========")
    group_output = locations.gtex_output / group
    group_output.mkdir(exist_ok=True, parents=True)
    eval_train= booster.eval(data = train_dataset)
    print(f"EVALUATION_TRAIN: \n{eval_train}")
    eval_test = booster.eval(data =test_dataset)
    print(f"EVALUATION_TEST: \n{eval_test}")
    par_output = (group_output / "params.json")
    par_output.touch(exist_ok=True)
    with par_output.open("w") as outfile:
        json.dump(best_params, outfile)
    print("TUNED!")
    return best_value, best_params


def tune_gtex(locations: Locations, death: str = None):
    print("tune_gtex, tuning best predictions for medium age")
    expressions = pl.read_parquet(locations.gtex_interim / "expressions_extended.parquet")
    to_predict = pl.col("medium_age")
    for_selection = [ensembl_gene_col, to_predict]
    death_types = {"ventilator_death": 0,
                   "fast_violent_death": 1,
                   "fast_natural_death": 2,
                   "ill_unexpected_death": 3,
                   "slow_chronic_death": 3}

    group = death_types[death] if death in death_types else "all"
    df = expressions.filter(pl.col("DTHHRDY")==death_types[death]) if death in death_types else expressions
    return tune_group(df, for_selection, group, locations)