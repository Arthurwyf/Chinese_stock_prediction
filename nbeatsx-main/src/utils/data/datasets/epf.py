__all__ = ['SOURCE_URL', 'NP', 'PJM', 'BE', 'FR', 'DE', 'EPFInfo', 'EPF']

import os
if not os.path.exists('./results/'):
    os.makedirs('./results/')

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .utils import download_file, Info, TimeSeriesDataclass

# Cell
SOURCE_URL = 'https://sandbox.zenodo.org/api/files/da5b2c6f-8418-4550-a7d0-7f2497b40f1b/'

# Cell
@dataclass
class NP:
    test_date: str = '2016-12-27'
    name: str = 'NP'

@dataclass
class PJM:
    test_date: str = '2016-12-27'
    name: str = 'PJM'

@dataclass
class BE:
    test_date: str = '2015-01-04'
    name: str = 'BE'

@dataclass
class FR:
    test_date: str = '2015-01-04'
    name: str = 'FR'

@dataclass
class DE:
    test_date: str = '2016-01-04'
    name: str = 'DE'

# Cell
EPFInfo = Info(groups=('NP', 'PJM', 'BE', 'FR', 'DE'),
               class_groups=(NP, PJM, BE, FR, DE))

# Cell
class EPF:

    @staticmethod
    def load(directory: str,
             group: str) -> Tuple[pd.DataFrame,
                                  Optional[pd.DataFrame],
                                  Optional[pd.DataFrame]]:
        """
        Downloads and loads EPF data.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        group: str
            Group name.
            Allowed groups: 'NP', 'PJM', 'BE', 'FR', 'DE'.
        """
        path = Path(directory) / 'epf' / 'datasets'

        EPF.download(directory)

        class_group = EPFInfo.get_group(group)

        file = path / f'{group}.csv'

        df = pd.read_csv(file)

        df.columns = ['ds', 'y'] + \
                     [f'Exogenous{i}' for i in range(1, len(df.columns) - 1)]

        df['unique_id'] = group
        df['ds'] = pd.to_datetime(df['ds'])
        df['week_day'] = df['ds'].dt.dayofweek

        dummies = pd.get_dummies(df['week_day'], prefix='day')
        df = pd.concat([df, dummies], axis=1)

        dummies_cols = [col for col in df if col.startswith('day')]

        Y = df.filter(items=['unique_id', 'ds', 'y'])
        X = df.filter(items=['unique_id', 'ds', 'Exogenous1', 'Exogenous2', 'week_day'] + \
                      dummies_cols)
        
        return Y, X, None

    @staticmethod
    def load_groups(directory: str,
                    groups: List[str]) -> Tuple[pd.DataFrame,
                                                Optional[pd.DataFrame],
                                                Optional[pd.DataFrame]]:
        """
        Downloads and loads panel of EPF data
        according of groups.

        Parameters
        ----------
        directory: str
            Directory where data will be downloaded.
        groups: List[str]
            Group names.
            Allowed groups: 'NP', 'PJM', 'BE', 'FR', 'DE'.
        """
        Y = []
        X = []
        for group in groups:
            Y_df, X_df, S_df = EPF.load(directory=directory, group=group)
            Y.append(Y_df)
            X.append(X_df)

        Y = pd.concat(Y).sort_values(['unique_id', 'ds']).reset_index(drop=True)
        X = pd.concat(X).sort_values(['unique_id', 'ds']).reset_index(drop=True)

        S = Y[['unique_id']].drop_duplicates().reset_index(drop=True)
        dummies = pd.get_dummies(S['unique_id'], prefix='static')
        S = pd.concat([S, dummies], axis=1)

        return Y, X, S

    @staticmethod
    def download(directory: str) -> None:
        """Downloads EPF Dataset."""
        path = Path(directory) / 'epf' / 'datasets'
        if not path.exists():
            for group in EPFInfo.groups:
                download_file(path, SOURCE_URL + f'{group}.csv')
    

    @staticmethod
    def load_n(directory: str) -> Tuple[pd.DataFrame,
                                                Optional[pd.DataFrame],
                                                Optional[pd.DataFrame]]:
        sheet_dict = pd.read_excel(directory, sheet_name=None)
        combined_df = pd.DataFrame()
        for sheet_name, df in sheet_dict.items():
            df['unique_id'] = df.columns.values[0]
            df.columns.values[0] = 'Time'
            combined_df = combined_df.append(df, ignore_index=True)
        # change time and other parameters combined_df['Time'] = combined_df['Time'].apply(lambda x: "".join(str(x).split("-")).split(" ")[0])
        combined_df = combined_df[['unique_id'] + [col for col in combined_df.columns if col != 'unique_id']]
        
        # 把中文去掉
        # new_column_names = ['unique_id', 'ds', 'KPJ', 'y', 'QKPJ', 'ZGJ', 'ZDJ', 'ZDF', 'CJL', 'CJE']
        new_column_names = ['unique_id', 'ds', 'Exogenous1', 'y', 'Exogenous2', 'Exogenous3', 'Exogenous4', 'Exogenous5', 'Exogenous6', 'Exogenous7']
        combined_df = combined_df.rename(columns=dict(zip(combined_df.columns, new_column_names)))

        combined_df['week_day'] = combined_df['ds'].dt.dayofweek
        dummies = pd.get_dummies(combined_df['week_day'], prefix='day')
        df = pd.concat([combined_df, dummies], axis=1)

        dummies_cols = [col for col in df if col.startswith('day')]
        # print(combined_df)
        # for predict 收盘价, 那么y就是 SPJ 
        Y = combined_df.filter(items=['unique_id', 'ds', 'y']) 
        X = combined_df.filter(items=['unique_id', 'ds', 'Exogenous1', 'Exogenous2', 'Exogenous3', 'Exogenous4', 'Exogenous5', 'Exogenous6', 'Exogenous7', 'week_day'] + dummies_cols)
        # print(Y, X)
        return Y, X, None


    @staticmethod
    def load_groups_n(directory, groups):
        csv_dir_path="/home/arthur/Time_Series_Forecasting/nbeatsx-main/new_dataset"
        Y = []
        X = []
        "load the data in a dir which contains multiple csv"
        for file in os.listdir(csv_dir_path):
            Y_df, X_df, S_df = EPF.load_n(os.path.join(csv_dir_path, file))
            Y.append(Y_df)
            X.append(X_df)
        # print(Y)
        # print(X)

        Y = pd.concat(Y).sort_values(['unique_id', 'ds']).reset_index(drop=True)
        X = pd.concat(X).sort_values(['unique_id', 'ds']).reset_index(drop=True)
        S = Y[['unique_id']].drop_duplicates().reset_index(drop=True)
        dummies = pd.get_dummies(S['unique_id'], prefix='static')
        S = pd.concat([S, dummies], axis=1)
        print(Y)
        print(X)
        print(S)
        return Y, X, S



if __name__ == "__main__":
    # EPF.load_n("/home/arthur/Time_Series_Forecasting/nbeatsx-main/new_dataset/材料类.xlsx")
    EPF.load_groups_n()
