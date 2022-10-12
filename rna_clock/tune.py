def tune_gtex(locations: Locations):
    print("train_gtex, prediction of medium age")
    coding_genes = pl.read_csv(locations.coding_human_genes, sep="\t")
    coding_gene_ids = coding_genes.select(pl.col("Gene stable ID")).to_dict(False)['Gene stable ID']
    expressions = pl.read_parquet(locations.gtex_interim / "expressions_extended.parquet")
    gene_columns = seq(expressions.columns).filter(lambda s: "ENSG" in s)
    coding_gene_columns = sorted(set(coding_gene_ids) & set(gene_columns), key = coding_gene_ids.index)
    to_predict: str = "medium_age"
    for_selection = coding_gene_columns + [to_predict]
    print("predicting with all samples")
    booster, metrics = split_and_train(expressions, for_selection)
    print("writing results metrics for all samples")
    BasicMetrics.write_json_array(metrics, locations.gtex_output / "gtex_metrics_plot.json")
    seq(metrics).min_by(lambda met: met.huber).write_json(locations.gtex_output / "gtex_metrics_best_huber.json")
    seq(metrics).min_by(lambda met: met.MSE).write_json(locations.gtex_output / "gtex_metrics_best_mse.json")
"""

@dataclass
class LightTuner(TransformerMixin):
    '''
    It is somewhat buggy, see https://github.com/optuna/optuna/issues/1602#issuecomment-670937574
    I had to switch to GeneralTuner while they are fixing it
    '''

    time_budget_seconds: int

    parameters: Dict = field(default_factory=lambda: {
        'boosting_type': 'dart',
        'objective': 'regression',
        'metric': 'huber'
    })
    num_boost_round: int = 500
    early_stopping_rounds = 5
    seed: int = 42

    def fit(self, partitions: ExpressionPartitions, y=None) -> Dict:
        cat = partitions.categorical_index if partitions.features.has_categorical else "auto"
        lgb_train = lgb.Dataset(partitions.X, partitions.Y, categorical_feature=cat, free_raw_data=False)
        tuner = optuna.integration.lightgbm.LightGBMTunerCV(
            self.parameters, lgb_train, verbose_eval=self.num_boost_round, folds=partitions.folds,
            time_budget=self.time_budget_seconds,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds
        )
        tuner.tune_bagging()
        tuner.tune_feature_fraction()
        tuner.tune_min_data_in_leaf()
        tuner.tune_feature_fraction_stage2()
        tuner.run()
        return SpecializedTuningResults(tuner.best_params, tuner.best_score)
"""
