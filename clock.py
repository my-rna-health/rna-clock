#!/usr/bin/env python
import urllib
import urllib.request
from pathlib import Path

import click
import polars as pl

data = Path("data")
input = data / "input"
gtex = input / "gtex"
gtex.mkdir(parents=True, exist_ok=True)
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
    output_file = gtex / "GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad"
    if skit_if_exist and output_file.exists():
        print("scRNA-Seq file exists, skipping downloading it")
    else:
        urllib.request.urlretrieve(url, gtex / "GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad")
    immune_url = "https://storage.googleapis.com/gtex_analysis_v9/snrna_seq_data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"
    immune_file = gtex / "GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"
    if skit_if_exist and immune_file.exists():
        print("skipping immune downloads")
    else:
        urllib.request.urlretrieve(immune_url , immune_file)

def stabilize_rna(path: Path) -> pl.DataFrame:
    rnas = pl.read_csv(path, comment_char="#", sep="\t")
    stable_gene = pl.col("Name").str.split(".").alias("Ensembl").apply(lambda s: s[0])
    return rnas.select([
        stable_gene,
        pl.all()
    ]).drop(["Name", "Description"])

def download_gtex_tpms(skip_if_exist: bool = True):
    url = "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    rna_name = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct"
    stable_rna_file = gtex / rna_name.replace("_tpm", "_tpm_stable")
    rna_file_gz = gtex / (rna_name + ".gz")
    rna_file = gtex / rna_name
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
    gct = str(gtex / rna_name)
    exp_name = rna_name.replace(".gct", "_expressions.tsv")
    exp_path = gtex / exp_name
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
    sample_file = gtex / "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
    urllib.request.urlretrieve(sample_url, sample_file)
    phenotypes_url = "https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
    phenotypes_file = gtex / "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
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


if __name__ == '__main__':
    app()