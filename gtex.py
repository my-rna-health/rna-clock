#!/usr/bin/env python
import urllib
import urllib.request

import click
import polars as pl
from pycomfort.files import *

data = Path("data")
gtex = data / "gtex"
gtex.mkdir(parents=True, exist_ok=True)

gtex_input = gtex / "input"
gtex_input.mkdir(parents=True, exist_ok=True)

gtex_interim = gtex / "interim"
gtex_interim.mkdir(parents=True, exist_ok=True)

proxy_handler = urllib.request.ProxyHandler({})
opener = urllib.request.build_opener(proxy_handler)
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

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
    output_file = gtex_input / "GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad"
    if skit_if_exist and output_file.exists():
        print("scRNA-Seq file exists, skipping downloading it")
    else:
        urllib.request.urlretrieve(url, output_file)
    immune_url = "https://storage.googleapis.com/gtex_analysis_v9/snrna_seq_data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"
    immune_file = gtex_input / "GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"
    if skit_if_exist and immune_file.exists():
        print("skipping immune downloads")
    else:
        urllib.request.urlretrieve(immune_url, immune_file)


def stabilize_rna(path: Path) -> pl.DataFrame:
    rnas = pl.read_csv(path, comment_char="#", sep="\t")
    stable_gene = pl.col("Name").str.split(".").alias("Ensembl").apply(lambda s: s[0]).unique()
    return rnas.select([
        stable_gene,
        pl.all()
    ]).drop(["Name", "Description"])


def download_gtex_tpms(skip_if_exist: bool = True):
    url = "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    rna_name = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"
    stable_rna_file = gtex_input / rna_name.replace("_tpm", "_tpm_stable")
    rna_file_gz = gtex_input / (rna_name + ".gz")
    rna_file = gtex_input / rna_name
    if skip_if_exist and stable_rna_file.exists():
        print("file exists, skipping")
    else:
        urllib.request.urlretrieve(url, rna_file_gz)
        un_gzip(rna_file_gz)
        rna_file_gz.unlink(missing_ok=True)
        with open(rna_file, 'r+') as fp:
            lines = fp.readlines()
            fp.seek(0)
            fp.truncate()
            fp.writelines(lines[2:])
            stable_rnas = stabilize_rna(rna_file)
            stable_rnas.write_csv(str(stable_rna_file), sep="\t")
            rna_file.unlink(missing_ok=True)
            print(f"produced stable rna file at {str(stable_rna_file)}")
            transpose_gtex_bulk(stable_rna_file.name, skip_if_exist)


def transpose_gtex_bulk(rna_name: str, skip_if_exist: bool = True):
    gct = str(gtex_interim / rna_name)
    exp_name = rna_name.replace(".gct", "_expressions.tsv")
    exp_path = gtex_interim / exp_name
    if skip_if_exist and exp_path.exists():
        print("skipping transposing")
    else:
        print("transposing gtex bulk rna-seq")
        import subprocess
        comm = f"datamash transpose < {gct} > {str(exp_path)}"
        print(f"command to run: {comm}")
        transpose = subprocess.run(comm, shell=True)
        print("The exit code was: %d" % transpose.returncode)


def download_gtex_sample(skip_if_exist: bool = True):
    print("downloading gtex samples info")
    sample_url = "https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    sample_file = gtex_input / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    urllib.request.urlretrieve(sample_url, sample_file)
    phenotypes_url = "https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
    phenotypes_file = gtex_input / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
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
    samples_with_subjects_path = gtex_interim / "samples_with_subjects.parquet"
    samples_with_subjects = pl.read_parquet(samples_with_subjects_path)
    return samples_with_subjects.join(exp, left_on="SAMPID", right_on="Ensembl")

@app.command()
def prepare_bulk():
    sample_file = gtex_input / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    phenotypes_file = gtex_input / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
    samples = pl.read_csv(sample_file, comment_char="#", sep="\t", ignore_errors=True)
    phenotypes = pl.read_csv(phenotypes_file, comment_char="#", sep="\t", ignore_errors=True)
    medium_age = pl.col("AGE").str.split("-").alias("medium_age").apply(lambda s: (int(s[1])+int(s[0]))/2.0)
    phenotypes_extended = phenotypes.with_column(medium_age)
    col_sample = pl.col("SAMPID")
    col_get_subject = col_sample.apply(lambda s: s.split("-")[0]+"-"+s.split("-")[1]).alias("SUBJID")
    samples_with_subjects = phenotypes_extended.join(samples.with_column(col_get_subject), left_on="SUBJID", right_on="SUBJID")
    #saving results
    samples_with_subjects_path = gtex_interim / "samples_with_subjects.parquet"
    samples_with_subjects.write_parquet(samples_with_subjects_path)
    print(f"files saved to {samples_with_subjects_path}")
    exp_path = gtex_interim / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm_stable_expressions.tsv"
    print(f"loading expressions from {exp_path}")
    expressions = pl.read_csv(str(exp_path), comment_char="#", sep="\t", ignore_errors=True)
    expressions_extended = with_subjects(expressions)
    expressions_extended.write_parquet(gtex_interim / "expressions_extended.parquet")


if __name__ == '__main__':
    app()