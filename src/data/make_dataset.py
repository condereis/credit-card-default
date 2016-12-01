# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
import scipy as sp

from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    raw = pd.read_csv(project_dir + '/data/raw/UCI_Credit_Card.csv')
    
    V = sp.std(raw['LIMIT_BAL'])
    raw['LIMIT_BAL'] = (raw['LIMIT_BAL'] - raw['LIMIT_BAL'].mean())/V
    raw['SEX'] = raw['SEX'] - 1;
    raw['EDUCATION'] = raw['EDUCATION'].replace(0,4);
    raw['EDUCATION'] = raw['EDUCATION'].replace(5,4);
    raw['EDUCATION']= raw['EDUCATION'].replace(6,4);
    raw['MARRIAGE'] = raw['MARRIAGE'].replace(2,0);
    raw['MARRIAGE'] = raw['MARRIAGE'].replace(3,0);
    V = sp.std(raw['AGE'])
    raw['AGE'] = (raw['AGE'] - raw['AGE'].mean())/V;
    V = sp.std(raw['BILL_AMT1'])
    raw['BILL_AMT1'] = (raw['BILL_AMT1'] - raw['BILL_AMT1'].mean())/V
    V = sp.std(raw['BILL_AMT2'])
    raw['BILL_AMT2'] = (raw['BILL_AMT2'] - raw['BILL_AMT2'].mean())/V
    V = sp.std(raw['BILL_AMT3'])
    raw['BILL_AMT3'] = (raw['BILL_AMT3'] - raw['BILL_AMT3'].mean())/V
    V = sp.std(raw['BILL_AMT4'])
    raw['BILL_AMT4'] = (raw['BILL_AMT4'] - raw['BILL_AMT4'].mean())/V
    v = sp.std(raw['BILL_AMT5'])
    raw['BILL_AMT5'] = (raw['BILL_AMT5'] - raw['BILL_AMT5'].mean())/V
    V = sp.std(raw['BILL_AMT6'])
    raw['BILL_AMT6'] = (raw['BILL_AMT6'] - raw['BILL_AMT6'].mean())/V
    V = sp.std(raw['PAY_AMT1'])
    raw['PAY_AMT1'] = (raw['PAY_AMT1'] - raw['PAY_AMT1'].mean())/V
    V = sp.std(raw['PAY_AMT2'])
    raw['PAY_AMT2'] = (raw['PAY_AMT2'] - raw['PAY_AMT2'].mean())/V
    V = sp.std(raw['PAY_AMT3'])
    raw['PAY_AMT3'] = (raw['PAY_AMT3'] - raw['PAY_AMT3'].mean())/V
    V = sp.std(raw['PAY_AMT4'])
    raw['PAY_AMT4'] = (raw['PAY_AMT4'] - raw['PAY_AMT4'].mean())/V
    V = sp.std(raw['PAY_AMT5'])
    raw['PAY_AMT5'] = (raw['PAY_AMT5'] - raw['PAY_AMT5'].mean())/V
    V = sp.std(raw['PAY_AMT6'])
    raw['PAY_AMT6'] = (raw['PAY_AMT6'] - raw['PAY_AMT6'].mean())/V

    raw['ED_1'] = (raw.EDUCATION == 1).astype(int)
    raw['ED_2'] = (raw.EDUCATION == 2).astype(int)
    raw['ED_3'] = (raw.EDUCATION == 3).astype(int)
    raw['ED_4'] = (raw.EDUCATION == 4).astype(int)
    raw.drop('EDUCATION',axis=1, inplace=True)
    
    
    train, test = train_test_split(raw, test_size=0.3)
    train.to_csv(project_dir + '/data/interim/train.csv', index=False)
    test.to_csv(project_dir + '/data/interim/test.csv', index=False)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    

main() 
