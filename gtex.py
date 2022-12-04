#!/usr/bin/env python
import ssl
import urllib
import urllib.request

import click
import polars as pl
from pycomfort.files import *

from rna_clock.train import train_gtex
from rna_clock.config import Locations
from rna_clock.tune import tune_gtex

locations = Locations(Path("..") if Path(".").name == "rna_clock" else Path("."))

proxy_handler = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(proxy_handler)
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)
try:
    ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = ssl._create_unverified_context

@click.group()
def app():
    print("running rna clock application")


def un_gzip(fl: Path):
    import gzip
    import shutil
    with gzip.open(str(fl), 'rb') as f_in:
        with open(str(fl).replace(".gz", ""), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


@app.command()
def clean():
    print("cleaning files")


def download_gtex_single_cell(skit_if_exist: bool = True):
    print("downloading single cell")
    url = "https://storage.googleapis.com/gtex_analysis_v9/snrna_seq_data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad"
    output_file = locations.gtex_input / "GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad"
    if skit_if_exist and output_file.exists():
        print("scRNA-Seq file exists, skipping downloading it")
    else:
        urllib.request.urlretrieve(url, output_file)
    immune_url = "https://storage.googleapis.com/gtex_analysis_v9/snrna_seq_data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"
    immune_file = locations.gtex_input / "GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"
    if skit_if_exist and immune_file.exists():
        print("skipping immune downloads")
    else:
        urllib.request.urlretrieve(immune_url, immune_file)


def stabilize_rna(path: Path) -> pl.DataFrame:
    print("preparing stable ids for RNA")
    stable_gene = pl.col("Name").str.split(".").alias("Ensembl").apply(lambda s: s[0])
    rnas = pl.read_csv(path , comment_char="#", sep="\t", skip_rows=2)
    return rnas.select([
        stable_gene, pl.all()
    ]).drop(["Name", "Description"]).groupby(pl.col("Ensembl")).agg(pl.all().sum())


def download_cattle_counts(skip_if_exist: bool = True):
    #url = "https://cgtex.roslin.ed.ac.uk/wp-content/plugins/cgtex/static/rawdata/Gene_read_counts_FarmGTEx_cattle_V0.txt.txt.gz"
    url = "/home/antonkulaga/Downloads/TPM.8742samples_27607genes.zip"
    #counts_name = "Gene_read_counts_FarmGTEx_cattle_V0.tsv"
    counts_name = "TPM.8742samples_27607genes.zip"
    counts_file_gz = locations.cattle_input / (counts_name + ".gz")
    counts_file = locations.cattle_input / counts_name
    if skip_if_exist and counts_file.exists():
        print("file exists, skipping")
    else:
        if not counts_file_gz.exists():
            urllib.request.urlretrieve(url, counts_file_gz)
            un_gzip(counts_file_gz)
            counts_file_gz.unlink(missing_ok=True)
            print("cattle download finished")


def download_gtex_tpms(skip_if_exist: bool = True):
    url = "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    rna_name = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"
    stable_rna_file = locations.gtex_input / rna_name.replace("_tpm", "_tpm_stable")
    rna_file_gz = locations.gtex_input / (rna_name + ".gz")
    rna_file = locations.gtex_input / rna_name
    if skip_if_exist and stable_rna_file.exists():
        print("file exists, skipping")
    else:
        if not rna_file_gz.exists():
            urllib.request.urlretrieve(url, rna_file_gz)
        un_gzip(rna_file_gz)
        rna_file_gz.unlink(missing_ok=True)
        #with open(rna_file, 'r+') as fp:
        #    lines = fp.readlines()
        #    fp.seek(0)
        #    fp.truncate()
        #    fp.writelines(lines[2:])
        stable_rnas = stabilize_rna(rna_file)
        stable_rnas.write_csv(str(stable_rna_file), sep="\t")
        rna_file.unlink(missing_ok=True)
        print(f"produced stable rna file at {str(stable_rna_file)}")
        transpose_gtex_bulk(stable_rna_file.name, skip_if_exist)


def transpose_gtex_bulk(rna_name: str, skip_if_exist: bool = True):
    print("transposing bulk gtex data")
    gct = str(locations.gtex_input / rna_name)
    exp_name = rna_name.replace(".gct", "_expressions.tsv")
    exp_path = locations.gtex_interim / exp_name
    if skip_if_exist and exp_path.exists():
        print("skipping transposing")
    else:
        print("transposing gtex bulk rna-seq")
        import subprocess
        comm = f"datamash transpose < {gct} > {str(exp_path)}"
        print(f"command to run: {comm}")
        transpose = subprocess.run(comm, shell=True)
        print("The exit code was: %d" % transpose.returncode)
        print(f"transpose of {exp_path.absolute().resolve()}")


def download_gtex_sample(skip_if_exist: bool = True):
    print("downloading gtex samples info")
    sample_url = "https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    sample_file = locations.gtex_input / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    urllib.request.urlretrieve(sample_url, sample_file)
    phenotypes_url = "https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
    phenotypes_file = locations.gtex_input / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
    urllib.request.urlretrieve(phenotypes_url, phenotypes_file)


def download_gtex_bulk(skip_if_exist: bool = True):
    print("downloading bulk")
    download_gtex_tpms(skip_if_exist)
    download_gtex_sample(skip_if_exist)

def download_gtex():
    print("downloading gtex files")
    download_gtex_single_cell()
    download_gtex_bulk()


@app.command()
def download():
    download_gtex()

#def with_subjects(exp: pl.DataFrame):
#    pl.col("my").str.split(",").arr.
#    pl.col("my_column").where()

def with_subjects(exp: pl.DataFrame):
    samples_with_subjects_path = locations.gtex_interim / "samples_with_subjects.parquet"
    samples_with_subjects = pl.read_parquet(samples_with_subjects_path)
    return samples_with_subjects.join(exp, left_on="SAMPID", right_on="Ensembl")

@app.command()
def prepare_bulk():
    sample_file = locations.gtex_input / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    phenotypes_file = locations.gtex_input / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
    samples = pl.read_csv(sample_file, comment_char="#", sep="\t", ignore_errors=True)
    phenotypes = pl.read_csv(phenotypes_file, comment_char="#", sep="\t", ignore_errors=True)
    medium_age = pl.col("AGE").str.split("-").alias("medium_age").apply(lambda s: (int(s[1])+int(s[0]))/2.0)
    phenotypes_extended = phenotypes.with_column(medium_age)
    col_sample = pl.col("SAMPID")
    col_get_subject = col_sample.apply(lambda s: s.split("-")[0]+"-"+s.split("-")[1]).alias("SUBJID")
    samples_with_subjects = phenotypes_extended.join(samples.with_column(col_get_subject), left_on="SUBJID", right_on="SUBJID")
    #saving results
    samples_with_subjects_path = locations.gtex_interim / "samples_with_subjects.parquet"
    samples_with_subjects.write_parquet(samples_with_subjects_path)
    print(f"files saved to {samples_with_subjects_path}")
    exp_path = locations.gtex_interim / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm_stable_expressions.tsv"
    print(f"loading expressions from {exp_path}")
    expressions = pl.read_csv(str(exp_path), comment_char="#", sep="\t", ignore_errors=True)
    expressions_extended = with_subjects(expressions)
    expressions_extended.write_parquet(locations.gtex_interim / "expressions_extended.parquet")

@app.command()
def prepare_cattle_bulk():
    download_cattle_counts()

@app.command()
def tune():
    tune_gtex(locations)


@app.command()
def train():
    train_gtex(locations)

if __name__ == '__main__':
    app()