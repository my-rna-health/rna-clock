from rna_clock.config import Locations
from rna_clock.splits import *
from rna_clock.trees import *


def split_and_train(expressions: pl.DataFrame,
                    for_selection: list[pl.col] = [pl.col("^ENSG[a-zA-Z0-9]+$"), pl.col("medium_age")],
                    num_boost_round: int = 1000,
                    early_stopping_rounds = 50) -> [Booster, list[BasicMetrics]]:
    df = expressions.select(for_selection)
    [train, test] = with_train_test_split(df) #with_train_test_eval_split(df)
    print(f"train [{train.shape}], test [{test.shape}]")
    (train_X, train_Y) = to_XY(train)
    (test_X, test_Y) = to_XY(test)
    return train_lightgbm_model(train_X, test_X, train_Y, test_Y,
                                validation_name="development",
                                num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds
                                )

def train_group(expressions: pl.DataFrame,
                for_selection: list[pl.col] = [pl.col("^ENSG[a-zA-Z0-9]+$"), pl.col("medium_age")],
                group: str = "all",
                locations: Locations = None,
                num_boost_round: int = 1000, early_stopping_rounds = 50):
    print(f"predicting with {group} samples")
    booster, metrics = split_and_train(expressions, for_selection, num_boost_round, early_stopping_rounds)
    print("writing results metrics for all samples")
    best_huber = seq(metrics).min_by(lambda met: met.huber)
    print(f"metrics with best huber are:\n", best_huber)
    best_mse = seq(metrics).min_by(lambda met: met.MSE)
    if locations is None:
        return booster, metrics
    else:
        group_output = locations.gtex_output / group
        group_output.mkdir(exist_ok=True)
        BasicMetrics.write_json_array(metrics, group_output / "gtex_metrics_plot.json")
        best_huber.write_json(group_output / "gtex_metrics_best_huber.json")
        best_mse.write_json(group_output / "gtex_metrics_best_mse.json")


def train_gtex(locations: Locations):
    print("train_gtex, prediction of medium age")
    coding_genes = pl.read_csv(locations.coding_human_genes, sep="\t")
    coding_gene_ids = coding_genes.select(pl.col("Gene stable ID")).to_dict(False)['Gene stable ID']
    expressions = pl.read_parquet(locations.gtex_interim / "expressions_extended.parquet")
    gene_columns = seq(expressions.columns).filter(lambda s: "ENSG" in s)
    coding_gene_columns = sorted(set(coding_gene_ids) & set(gene_columns), key = coding_gene_ids.index)
    to_predict: str = "medium_age"
    for_selection = coding_gene_columns + [to_predict]
    train_group(expressions, for_selection, "all", locations)




