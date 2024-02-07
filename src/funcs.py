"""
Módulo contendo funções de processamento de dados
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def check_nan(
        df: pd.DataFrame = None,
        cols: list = None,
    ):
    """
    Resumo dos dados para valores ausentes

    Args:
        df (pd.DataFrame):
        Dataframe a ser analisado

        cols (list):
        colunas a serem analisadas

    Returns:
        pd.DataFrame: DataFrame com a quantidade de valores ausentes por coluna no DataFrame
    """

    dic_nan = {'feature':[], 'type':[], 'missing_total':[], 'percentage %':[]}
    total = df.shape[0]
    for col in df.columns:
        sum_na = df[col].isna().sum()
        dic_nan['feature'].append(col)
        dic_nan['type'].append(df[col].dtypes)
        dic_nan['missing_total'].append(sum_na)
        dic_nan['percentage %'].append(round((sum_na/total) * 100, 3))
    return pd.DataFrame(dic_nan).sort_values(by='missing_total', ascending=False)

def cat_encoder(
        df: pd.DataFrame = None,
        cols_enc: list = None
    ):
    """
    Encoding de variáveis categóricas

    Args:
        df (pd.DataFrame):
        Dataframe que será utilizado para realizar a codificação das respectivas variáveis categóricas

        cols_enc (list):
        Uma lista contendo o nome das colunas que serão codificadas

    Returns:
        pd.DataFrame: Dataframe com as colunas categóricas codificadas
    """
    df_novo = df.copy()

    encoder = LabelEncoder()
    enc_mapping = list()

    for col in cols_enc:
        encoded = encoder.fit_transform(df_novo[col].astype(str))
        df_novo[col] = encoded
        df_novo[col] = df_novo[col].values.astype('int64')
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        enc_mapping.append([col, mapping])

    return df_novo

def norm_features(
        df: pd.DataFrame = None,
        except_cols: list = None
    ):
    """
    Normalizando variáveis numéricas

    Args:
        df (pd.DataFrame):
        Dataframe que será utilizado para realizar a normalização das variáveis numéricas

        except_cols (list):
        Lista contendo o nome das colunas que serão excluídas do df

    Returns:
        pd.DataFrame: Dataframe com as colunas numéricas normalizadas.
    """
    df_novo = df.copy()

    num_cols = [x for x in df_novo.select_dtypes(include=['int32' ,'int64', 'float64']).columns]
    cols_norm = list(set(num_cols) - set(except_cols))

    scaler = MinMaxScaler()

    for coluna in cols_norm:
        encoded = scaler.fit_transform(df_novo[coluna].values.reshape(-1, 1))
        df_novo[coluna] = encoded

    return df_novo