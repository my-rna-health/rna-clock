#!/usr/bin/env python
import urllib
import urllib.request
import zipfile
from pathlib import Path

import click

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
    import shutil


def download_single_cell(skit_if_exist: bool = True):
    print("downloading single cell")
    url = "https://storage.googleapis.com/gtex_analysis_v9/snrna_seq_data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad"
    output_file = gtex / "GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad"
    if skit_if_exist and output_file.exists():
        print("scRNA-Seq file exists")
    else:
        urllib.request.urlretrieve(url, gtex / "GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad")
    immune_url = "https://storage.googleapis.com/gtex_analysis_v9/snrna_seq_data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"
    immune_file = gtex / "GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad"
    if skit_if_exist and immune_file.exists():
        print("skipping downloads")
    else:
        urllib.request.urlretrieve(immune_url , immune_file)


def download_bulk(skit_if_exist: bool = True):
    print("downloading bulk")
    url = "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    rna_file = gtex / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
    if skit_if_exist and rna_file.exists():
        print("file exists, skipping")
    else:
        urllib.request.urlretrieve(url, rna_file)
    un_gzip(rna_file)
    rna_file.unlink(missing_ok=True)


@app.command()
def download():
    print("downloading gtex files")
    download_single_cell()
    download_bulk()

if __name__ == '__main__':
    app()