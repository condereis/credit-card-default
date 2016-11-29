# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd

from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw = pd.read_csv(project_dir + '/data/raw/UCI_Credit_Card.csv')
    train, test = train_test_split(raw, test_size=0.3)
    train.to_csv(project_dir + '/data/interim/train.csv')
    test.to_csv(project_dir + '/data/interim/test.csv')
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    

    main()
