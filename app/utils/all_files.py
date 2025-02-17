# @title
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


class DataFrameProcessor:
    def __init__(self, df):
        self.df = df.copy()
        # Features pour DecisionTree
        self.tree_features = [
            'Année BAC', 'Nbre Fois au BAC', 'Groupe Résultat', 'Moy. nde',
            'Moy. ère', 'Moy. S Term.', 'Moy. S Term..1', 'MATH', 'SCPH', 'FR',
            'PHILO', 'AN', 'Tot. Pts au Grp.', 'Moyenne au Grp.', 'Moy. Gle',
            'Moy. sur Mat.Fond.', 'Age en Décembre 2018', 'Sexe_F', 'Sexe_M',
            'Série_S1', 'Série_S2', 'Série_S3', 'Mention_ABien', 'Mention_Bien',
            'Mention_Pass', 'Résidence', 'Ets. de provenance', 'Centre d\'Ec.',
            'Académie de l\'Ets. Prov.', 'REGION_DE_NAISSANCE', 'Academie perf.'
        ]

    def preprocess_data_tree(self):
        """Prétraitement pour les DecisionTrees"""
        df = self.df.copy()

        # One-hot encoding pour colonnes catégorielles
        if 'Sexe' in df.columns:
            df['Sexe_F'] = (df['Sexe'] == 'F').astype(int)
            df['Sexe_M'] = (df['Sexe'] == 'M').astype(int)
            df.drop('Sexe', axis=1, inplace=True)

        if 'Série' in df.columns:
            df['Série_S1'] = (df['Série'] == 'S1').astype(int)
            df['Série_S2'] = (df['Série'] == 'S2').astype(int)
            df['Série_S3'] = (df['Série'] == 'S3').astype(int)
            df.drop('Série', axis=1, inplace=True)

        if 'Mention' in df.columns:
            df['Mention_Pass'] = (df['Mention'] == 'Passable').astype(int)
            df['Mention_ABien'] = (df['Mention'] == 'Assez-Bien').astype(int)
            df['Mention_Bien'] = (df['Mention'] == 'Bien').astype(int)
            df.drop('Mention', axis=1, inplace=True)

        # Label encoding pour colonnes catégorielles restantes
        categorical_cols = [
            'Résidence', 'Ets. de provenance', 'Centre d\'Ec.',
            'Académie de l\'Ets. Prov.', 'REGION_DE_NAISSANCE'
        ]

        le = LabelEncoder()
        for col in categorical_cols:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))

        # Calcul performance académique
        if 'Académie de l\'Ets. Prov.' in df.columns and 'Moy. Gle' in df.columns:
            academie_mean = df.groupby('Académie de l\'Ets. Prov.')['Moy. Gle'].mean()
            df['Academie perf.'] = df['Académie de l\'Ets. Prov.'].map(academie_mean)

        # Assurer présence de toutes les colonnes
        for col in self.tree_features:
            if col not in df.columns:
                df[col] = 0

        # Conversion en numérique et gestion NaN
        df = df[self.tree_features].apply(pd.to_numeric, errors='coerce').fillna(0)

        return df

    def preprocess_data_lasso(self, features_needed):
        """Prétraitement pour modèles Lasso"""
        df = self.df.copy()

        # print("Features attendues:", features_needed)
        # print("Colonnes disponibles:", df.columns.tolist())

        # Encodage des séries pour S1
        if 'Série' in df.columns:
            df['S1'] = (df['Série'] == 'S1').astype(int)
        elif 'Série_S1' in df.columns:
            df['S1'] = df['Série_S1']

        # Garder les colonnes qui existent déjà
        existing_features = ['MATH', 'SCPH', 'FR']
        for col in existing_features:
            if col not in df.columns:
                print(f"Colonne manquante: {col}")

        # Calcul des performances
        if 'Académie de l\'Ets. Prov.' in df.columns:
            academie_mean = df.groupby('Académie de l\'Ets. Prov.')['Moy. Gle'].mean()
            df['Academie perf.'] = df['Académie de l\'Ets. Prov.'].map(academie_mean)

        if 'Résidence' in df.columns:
            residence_mean = df.groupby('Résidence')['Moy. Gle'].mean()
            df['Residence perf.'] = df['Résidence'].map(residence_mean)

        # print("Colonnes après traitement:", df.columns.tolist())

        # Créer les colonnes manquantes
        for feature in features_needed:
            if feature not in df.columns:
                print(f"Création colonne manquante: {feature}")
                df[feature] = 0

        # Retourner dans le bon ordre
        return df[list(features_needed)].apply(pd.to_numeric, errors='coerce').fillna(0)


class DioresPredictorEnsemblisteLasso(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # self.dt_paths = {
        #     'admission': './app/data/Models/V2/admi_non_admi_best_model_DecisionTree.pkl',
        #     'session': './app/data/Models/V2/session_best_model_DecisionTree.pkl',
        #     'mention': './app/data/Models/V2/mention_best_model_DecisionTree.pkl'
        # }
        #
        # self.lasso_base_paths = {
        #     'non_admi': './app/data/Models/Lasso_Admi_Session/NON_ADMI/non_admi/',
        #     'deuxieme_session': './app/data/Models/Lasso_Admi_Session/DEUXIME_SESSION/deuxieme_session/',
        #     'passable': './app/dataModels/Lasso_Admi_Session/PASSABLE/passable/',
        #     'mention': './app/data/Models/Lasso_Admi_Session/MENTION/mention/'
        # }



        # Obtenir le chemin absolu du répertoire contenant le script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(app_dir, "app", "data", "Models")

        # Définir les chemins pour les modèles DecisionTree
        self.dt_paths = {
            'admission': os.path.join(base_path, "V2", "admi_non_admi_best_model_DecisionTree.pkl"),
            'session': os.path.join(base_path, "V2", "session_best_model_DecisionTree.pkl"),
            'mention': os.path.join(base_path, "V2", "mention_best_model_DecisionTree.pkl")
        }

        # Définir les chemins pour les modèles Lasso
        lasso_base = os.path.join(base_path, "Lasso_Admi_Session")
        self.lasso_base_paths = {
            'non_admi': os.path.join(lasso_base, "NON_ADMI", "non_admi"),
            'deuxieme_session': os.path.join(lasso_base, "DEUXIME_SESSION", "deuxieme_session"),
            'passable': os.path.join(lasso_base, "PASSABLE", "passable"),
            'mention': os.path.join(lasso_base, "MENTION", "mention")
        }

        self.dt_models = {}
        self.lasso_models = {}
        self.lasso_scalers = {}
        self.lasso_features = {}

        self.load_models()

    def load_models(self):
        # Chargement DecisionTrees
        for key, path in self.dt_paths.items():
            with open(path, 'rb') as f:
                self.dt_models[key] = pickle.load(f)

        # Chargement modèles Lasso
        for key, base_path in self.lasso_base_paths.items():
            try:
                with open(f"{base_path}/{key}_lasso_model.pkl", 'rb') as f:
                    self.lasso_models[key] = pickle.load(f)
                with open(f"{base_path}/{key}_lasso_scaler.pkl", 'rb') as f:
                    self.lasso_scalers[key] = pickle.load(f)
                with open(f"{base_path}/{key}_lasso_info.pkl", 'rb') as f:
                    self.lasso_features[key] = pickle.load(f)['features']
            except Exception as e:
                print(f"Erreur chargement modèle {key}: {str(e)}")

    def predict_student(self, X):
        try:
            # Prédiction admission avec DecisionTree
            X_tree = DataFrameProcessor(X).preprocess_data_tree()
            admission_pred = self.dt_models['admission'].predict(X_tree)[0]

            if admission_pred == 0:  # NON ADMIS
                X_lasso = DataFrameProcessor(X).preprocess_data_lasso(self.lasso_features['non_admi'])
                X_scaled = self.lasso_scalers['non_admi'].transform(X_lasso)
                score = self.lasso_models['non_admi'].predict(X_scaled)[0]
                return {'status': 'NON ADMIS', 'score': score, 'model': 'non_admi'}

            # Prédiction session
            session_pred = self.dt_models['session'].predict(X_tree)[0]

            if session_pred == 0:  # DEUXIÈME SESSION
                X_lasso = DataFrameProcessor(X).preprocess_data_lasso(self.lasso_features['deuxieme_session'])
                X_scaled = self.lasso_scalers['deuxieme_session'].transform(X_lasso)
                score = self.lasso_models['deuxieme_session'].predict(X_scaled)[0]
                return {'status': 'DEUXIÈME SESSION', 'score': score, 'model': 'deuxieme_session'}

            # Prédiction mention
            mention_pred = self.dt_models['mention'].predict(X_tree)[0]

            if mention_pred == 0:  # PASSABLE
                X_lasso = DataFrameProcessor(X).preprocess_data_lasso(self.lasso_features['passable'])
                X_scaled = self.lasso_scalers['passable'].transform(X_lasso)
                score = self.lasso_models['passable'].predict(X_scaled)[0]
                return {'status': 'PASSABLE', 'score': score, 'model': 'passable'}
            else:
                X_lasso = DataFrameProcessor(X).preprocess_data_lasso(self.lasso_features['mention'])
                X_scaled = self.lasso_scalers['mention'].transform(X_lasso)
                score = self.lasso_models['mention'].predict(X_scaled)[0]
                return {'status': 'MENTION SUPÉRIEURE', 'score': score, 'model': 'mention'}

        except Exception as e:
            print(f"Erreur prédiction: {str(e)}")
            raise
            'DEUXIÈME SESSION'
            'PASSABLE'
            'MENTION SUPÉRIEURE'

    def predict(self, X):
        """
        Prédit pour un ensemble d'étudiants et retourne le DataFrame original
        avec les colonnes de prédiction ajoutées

        Parameters:
        -----------
        X : pandas.DataFrame
            DataFrame contenant les données des étudiants

        Returns:
        --------
        pandas.DataFrame
            DataFrame original avec les colonnes de prédiction ajoutées
        """
        # Garder une copie du DataFrame original
        results_df = X.copy()

        predictions = []
        for idx, student in X.iterrows():
            try:
                pred = self.predict_student(pd.DataFrame([student]))
                predictions.append(pred)
            except Exception as e:
                print(f"Erreur de prédiction: {str(e)}")
                predictions.append({
                    'status': 'ERREUR',
                    'score': None,
                    'model': None
                })

        # Ajouter les colonnes de prédiction au DataFrame original
        results_df['Prediction_Status'] = [p['status'] for p in predictions]
        results_df['Score_Predit'] = [p['score'] for p in predictions]
        results_df['Model_Utilise'] = [p['model'] for p in predictions]

        return results_df


# ===================================================================


# @title
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


# # @title
# def evaluate_predictions(df_result, rank_method='average'):
#     df = df_result.copy()
#     df['Rang_L1_New'] = df['Score L1'].rank(ascending=False, method=rank_method)
#
#     #score_adjustments = {
#     #    'DEUXIÈME SESSION': -1000,
#     #    'PASSABLE': -5000,
#     #    'MENTION SUPÉRIEURE': -10000
#     #}
#     #for status, adjustment in score_adjustments.items():
#     #   df.loc[df['Prediction_Status'] == status, 'Score_Predit'] += adjustment
#
#     df['Rang_Predit'] = df['Score_Predit'].rank(ascending=False, method=rank_method)
#     df = df[['Rang_L1_New', 'Rang_Predit', 'RESULTAT']]
#
#     #rmse = np.sqrt(mean_squared_error(df['Rang_L1_New'], df['Rang_Predit']))
#     rmse = np.sqrt(root_mean_squared_error(df['Rang_L1_New'], df['Rang_Predit']))
#
#     # df['diff'] = (df['Rang_L1_New'] - df['Rang_Predit']).pow(2)
#     # rmse = np.sqrt(np.mean(df['diff']))
#     # print(f"RMSE_{suffix} : {rmse}")
#
#     df_strict = df.loc[df['RESULTAT'] == 'PASSE']
#     df_strict['mrr'] = 1 / df_strict['Rang_Predit']
#     mrr_strict = df_strict['mrr'].sum()
#     # print(f"MRR_{suffix} strict : {mrr_strict}")
#
#     df_open = df.loc[df['RESULTAT'] != 'NON ADMIS']
#     df_open['mrr'] = 1 / df_open['Rang_Predit']
#     mrr_open = df_open['mrr'].sum()
#     # print(f"MRR_{suffix} open : {mrr_open}")
#
#     return rmse, mrr_strict, mrr_open

def evaluate_predictions(df_result, rank_method='average'):
    df = df_result.copy()
    df['Rang_L1_New'] = df['Score L1'].rank(ascending=False, method=rank_method)
    df['Rang_Predit'] = df['Score_Predit'].rank(ascending=False, method=rank_method)
    df = df[['Rang_L1_New', 'Rang_Predit', 'RESULTAT']]

    rmse = np.sqrt(root_mean_squared_error(df['Rang_L1_New'], df['Rang_Predit']))

    # Créer des copies explicites pour df_strict et df_open
    df_strict = df.loc[df['RESULTAT'] == 'PASSE'].copy()
    df_open = df.loc[df['RESULTAT'] != 'NON ADMIS'].copy()

    # Utiliser .loc pour assigner les nouvelles valeurs
    df_strict.loc[:, 'mrr'] = 1 / df_strict['Rang_Predit']
    df_open.loc[:, 'mrr'] = 1 / df_open['Rang_Predit']

    mrr_strict = df_strict['mrr'].sum()
    mrr_open = df_open['mrr'].sum()

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


# Obtenir le chemin absolu du répertoire contenant le script
script_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.dirname(os.path.dirname(script_dir))
directory = os.path.join(app_dir, "app", "data", "Datasets", "L1MPI")

# Charger les fichiers
doc1_df_test = load_data(os.path.join(directory, "doc1", "doc1_df_test.csv"))
doc2_df_test = load_data(os.path.join(directory, "doc2", "doc2_df_test.csv"))
doc3_df_test = load_data(os.path.join(directory, "doc3", "doc3_df_test.csv"))

# Ajouter à la liste comme avant
docs = []
docs.append(doc1_df_test)
docs.append(doc2_df_test)
docs.append(doc3_df_test)

# directory = "./app/data/Datasets/L1MPI/"
# doc1_df_test = load_data(os.path.join(directory, "doc1", "doc1_df_test.csv"));
# doc2_df_test = load_data(os.path.join(directory, "doc2", "doc2_df_test.csv"));
# doc3_df_test = load_data(os.path.join(directory, "doc3", "doc3_df_test.csv"));
#
# docs = []
# docs.append(doc1_df_test)
# docs.append(doc2_df_test)
# docs.append(doc3_df_test)

predictor = DioresPredictorEnsemblisteLasso()
predictors = [predictor, predictor, predictor]
evaluate(docs, predictors, suffix='DIORES', rank_method='average')