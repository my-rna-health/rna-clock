#!/usr/bin/env python

from pathlib import Path

import click


@click.group()
def app():
    print("running cloning application")

@app.command()
def clean():
    print("cleaning files")
    import shutil
    shutil.rmtree('data/algae/MoCloKit', ignore_errors=True)
    shutil.rmtree('data/plant/MoClo', ignore_errors=True)