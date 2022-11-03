from rna_clock.config import Locations
from rna_clock.splits import *
from rna_clock.trees import *


def split_and_train(expressions: pl.DataFrame, for_selection: list[str]) -> [Booster, list[BasicMetrics]]:
    df = expressions.select(for_selection)
    [train, dev, test] = with_train_test_eval_split(df)
    print(f"train [{train.shape}], dev [{dev.shape}], test [{test.shape}]")
    (train_X, train_Y) = to_XY(train)
    (dev_X, dev_Y) = to_XY(train)
    return train_lightgbm_model(train_X, dev_X, train_Y, dev_Y, validation_name="development")


def train_group(expressions: pl.DataFrame, for_selection: list[str], group: str, locations: Locations):
    print("predicting with all samples")
    booster, metrics = split_and_train(expressions, for_selection)
    print("writing results metrics for all samples")
    group_output = locations.gtex_output / group
    group_output.mkdir(exist_ok=True)
    BasicMetrics.write_json_array(metrics, group_output / "gtex_metrics_plot.json")
    seq(metrics).min_by(lambda met: met.huber).write_json(group_output / "gtex_metrics_best_huber.json")
    seq(metrics).min_by(lambda met: met.MSE).write_json(group_output / "gtex_metrics_best_mse.json")


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




