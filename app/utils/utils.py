# @title
import os

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder


def oneHotEncoding(df, columnName):
    le = LabelEncoder()
    label = le.fit_transform(df[columnName])
    df.drop(columnName, axis=1, inplace=True)
    df[columnName] = label
    return df


def load_data(filePath):
    df = pd.read_csv(filePath)
    #for column in [ 'SEXE', 'Résidence', "Académie de l'Ets. Prov.", "Série", "Mention" ]:
    #  oneHotEncoding(df, column)
    return preprocess_data(df)


directory = "./data/Datasets/L1MPI/"
print("Chemin du fichier :", os.path.abspath(os.path.join(directory, "doc1", "doc1_df_test.csv")))
# directory = "/content/drive/MyDrive/Memoire/DIORES/Datasets/L1MPI/"


# doc1_df_test = load_data(os.path.join(directory, "doc1", "doc1_df_test.csv"));
# doc2_df_test = load_data(os.path.join(directory, "doc2", "doc2_df_test.csv"));
# doc3_df_test = load_data(os.path.join(directory, "doc3", "doc3_df_test.csv"));

# doc1_df_test = load_data(os.path.normpath(os.path.join(directory, "doc1", "doc1_df_test.csv")))


doc1_df_test = load_data(os.path.normpath(os.path.join(directory, "doc1", "doc1_df_test.csv")));
doc2_df_test = load_data(os.path.normpath(os.path.join(directory, "doc2", "doc2_df_test.csv")));
doc3_df_test = load_data(os.path.normpath(os.path.join(directory, "doc3", "doc3_df_test.csv")));

docs = []
docs.append(doc1_df_test)
docs.append(doc2_df_test)
docs.append(doc3_df_test)


def preprocess_data(df):
    """
    Effectue le prétraitement des données sur le DataFrame donné.

    Args:
        df (pd.DataFrame): Le DataFrame à prétraiter.

    Returns:
        pd.DataFrame: Le DataFrame prétraité.
    """
    # Remplacer les valeurs manquantes par 0
    df = df.fillna(0)

    # Encodage des séries
    def encode_series(df):
        series = ['S1', 'S2', 'S3']
        for serie in series:
            df[serie] = df['Série'].apply(lambda x: 1 if x == serie else 0)
        df.drop('Série', axis=1, inplace=True)
        return df

    # Encodage du sexe
    def encode_sexe(df):
        df['Homme'] = df['Sexe'].apply(lambda x: 1 if x == 'M' else 0)
        df['Femme'] = df['Sexe'].apply(lambda x: 1 if x == 'F' else 0)
        df.drop('Sexe', axis=1, inplace=True)
        return df

    # Moyennes par Académie
    def encode_academie_performance(df):
        academie_mean = df.groupby("Académie de l'Ets. Prov.")['Moy. Gle'].mean().to_dict()
        df['Academie perf.'] = df["Académie de l'Ets. Prov."].map(academie_mean)
        df.drop("Académie de l'Ets. Prov.", axis=1, inplace=True)
        return df

    # Moyennes par Résidence
    def encode_residence_performance(df):
        residence_mean = df.groupby("Résidence")['Moy. Gle'].mean().to_dict()
        df['Residence perf.'] = df["Résidence"].map(residence_mean)
        df.drop("Résidence", axis=1, inplace=True)
        return df

    # Conversion des colonnes non numériques en numériques
    def convert_non_numeric_columns(df):
        non_numeric_cols = df.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass
        return df

    # Supprimer la colonne "Mention"
    def drop_columns(df, columns):
        df.drop(columns, axis=1, inplace=True)
        return df

    # Appel des sous-fonctions
    df = encode_series(df)
    df = encode_sexe(df)
    df = encode_academie_performance(df)
    df = encode_residence_performance(df)
    df = drop_columns(df, ["Mention"])
    # df = convert_non_numeric_columns(df)

    return df


# @title
def evaluate_predictions(df_result, rank_method='average'):
    df = df_result.copy()
    df['Rang_L1_New'] = df['Score L1'].rank(ascending=False, method=rank_method)

    #score_adjustments = {
    #    'DEUXIÈME SESSION': -1000,
    #    'PASSABLE': -5000,
    #    'MENTION SUPÉRIEURE': -10000
    #}
    #for status, adjustment in score_adjustments.items():
    #   df.loc[df['Prediction_Status'] == status, 'Score_Predit'] += adjustment

    df['Rang_Predit'] = df['Score_Predit'].rank(ascending=False, method=rank_method)
    df = df[['Rang_L1_New', 'Rang_Predit', 'RESULTAT']]

    #rmse = np.sqrt(mean_squared_error(df['Rang_L1_New'], df['Rang_Predit']))
    rmse = np.sqrt(root_mean_squared_error(df['Rang_L1_New'], df['Rang_Predit']))

    # df['diff'] = (df['Rang_L1_New'] - df['Rang_Predit']).pow(2)
    # rmse = np.sqrt(np.mean(df['diff']))
    # print(f"RMSE_{suffix} : {rmse}")

    df_strict = df.loc[df['RESULTAT'] == 'PASSE']
    df_strict['mrr'] = 1 / df_strict['Rang_Predit']
    mrr_strict = df_strict['mrr'].sum()
    # print(f"MRR_{suffix} strict : {mrr_strict}")

    df_open = df.loc[df['RESULTAT'] != 'NON ADMIS']
    df_open['mrr'] = 1 / df_open['Rang_Predit']
    mrr_open = df_open['mrr'].sum()
    # print(f"MRR_{suffix} open : {mrr_open}")

    return rmse, mrr_strict, mrr_open


# # @title
def evaluate_all_predictions(df_results, suffix, rank_method='average'):
    print(f"EVALUATIONS POUR {suffix}")
    qualities = (0, 0, 0)
    for idx, df_result in enumerate(df_results):
        rmse, mrr_strict, mrr_open = evaluate_predictions(df_result, rank_method)
        qualities = np.add(qualities, (rmse, mrr_strict, mrr_open))
    qualities = qualities / len(df_results)
    print(f"RMSE_{suffix}_{rank_method} : {qualities[0]}")
    print(f"MRR_{suffix}_{rank_method} strict : {qualities[1]}")
    print(f"MRR_{suffix}_{rank_method} open : {qualities[2]}")


# @title
def evaluate(docs, predictors, suffix='', rank_method='average'):
    results = []
    for i in range(len(docs)):
        results.append(predictors[i].predict(docs[i].copy()))

    evaluate_all_predictions(results, suffix, rank_method)
