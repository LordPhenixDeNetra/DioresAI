import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import InconsistentVersionWarning

# Ignorer les avertissements de version de sklearn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

class DataFrameProcessor:
    """
    Classe pour le prétraitement des données avant la prédiction.
    Gère la transformation des features pour les modèles DecisionTree et Lasso.
    """

    def __init__(self, df):
        """
        Initialise le processeur avec un DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame à prétraiter
        """
        self.df = df.copy()
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
        """
        Prétraite les données pour les modèles DecisionTree.
        
        Returns:
            pd.DataFrame: DataFrame prétraité pour DecisionTree
        """
        df = self.df.copy()

        # One-hot encoding pour colonnes catégorielles
        encodings = {
            'Sexe': [('Sexe_F', 'F'), ('Sexe_M', 'M')],
            'Série': [('Série_S1', 'S1'), ('Série_S2', 'S2'), ('Série_S3', 'S3')],
            'Mention': [('Mention_Pass', 'Passable'),
                        ('Mention_ABien', 'Assez-Bien'),
                        ('Mention_Bien', 'Bien')]
        }

        for col, mappings in encodings.items():
            if col in df.columns:
                for new_col, value in mappings:
                    df[new_col] = (df[col] == value).astype(int)
                df.drop(col, axis=1, inplace=True)

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

        # Assurer présence de toutes les colonnes et conversion
        for col in self.tree_features:
            if col not in df.columns:
                df[col] = 0

        return df[self.tree_features].apply(pd.to_numeric, errors='coerce').fillna(0)

    def preprocess_data_lasso(self, features_needed):
        """
        Prétraite les données pour les modèles Lasso.

        Args:
            features_needed (list): Liste des features nécessaires pour le modèle

        Returns:
            pd.DataFrame: DataFrame prétraité pour Lasso
        """
        df = self.df.copy()

        # Encodage des séries
        if 'Série' in df.columns:
            df['S1'] = (df['Série'] == 'S1').astype(int)
        elif 'Série_S1' in df.columns:
            df['S1'] = df['Série_S1']

        # Calcul des performances
        if 'Académie de l\'Ets. Prov.' in df.columns:
            academie_mean = df.groupby('Académie de l\'Ets. Prov.')['Moy. Gle'].mean()
            df['Academie perf.'] = df['Académie de l\'Ets. Prov.'].map(academie_mean)

        if 'Résidence' in df.columns:
            residence_mean = df.groupby('Résidence')['Moy. Gle'].mean()
            df['Residence perf.'] = df['Résidence'].map(residence_mean)

        # Création des colonnes manquantes
        for feature in features_needed:
            if feature not in df.columns:
                df[feature] = 0

        return df[list(features_needed)].apply(pd.to_numeric, errors='coerce').fillna(0)


class DioresPredictorEnsemblisteLasso(BaseEstimator, ClassifierMixin):
    """
    Classificateur ensembliste combinant DecisionTrees et modèles Lasso
    pour la prédiction de la performance des étudiants.
    """

    def __init__(self):
        """Initialise le prédicteur avec les chemins des modèles."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(app_dir, "app", "data", "Models")

        # Chemins des modèles DecisionTree
        self.dt_paths = {
            'admission': os.path.join(base_path, "V2", "admi_non_admi_best_model_DecisionTree.pkl"),
            'session': os.path.join(base_path, "V2", "session_best_model_DecisionTree.pkl"),
            'mention': os.path.join(base_path, "V2", "mention_best_model_DecisionTree.pkl")
        }

        # Chemins des modèles Lasso
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
        """Charge tous les modèles nécessaires."""
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
        """
        Prédit le statut et le score d'un étudiant.

        Args:
            X (pd.DataFrame): DataFrame contenant les données d'un étudiant

        Returns:
            dict: Dictionnaire contenant le statut, le score et le modèle utilisé
        """
        try:
            X_tree = DataFrameProcessor(X).preprocess_data_tree()
            admission_pred = self.dt_models['admission'].predict(X_tree)[0]

            if admission_pred == 0:  # NON ADMIS
                return self._predict_with_lasso(X, 'non_admi', 'NON ADMIS')

            session_pred = self.dt_models['session'].predict(X_tree)[0]
            if session_pred == 0:  # DEUXIÈME SESSION
                return self._predict_with_lasso(X, 'deuxieme_session', 'DEUXIÈME SESSION')

            mention_pred = self.dt_models['mention'].predict(X_tree)[0]
            if mention_pred == 0:  # PASSABLE
                return self._predict_with_lasso(X, 'passable', 'PASSABLE')
            else:
                return self._predict_with_lasso(X, 'mention', 'MENTION SUPÉRIEURE')

        except Exception as e:
            print(f"Erreur prédiction: {str(e)}")
            raise

    def _predict_with_lasso(self, X, model_key, status):
        """
        Méthode utilitaire pour la prédiction avec un modèle Lasso spécifique.

        Args:
            X (pd.DataFrame): Données de l'étudiant
            model_key (str): Clé du modèle Lasso à utiliser
            status (str): Statut à retourner

        Returns:
            dict: Résultat de la prédiction
        """
        X_lasso = DataFrameProcessor(X).preprocess_data_lasso(self.lasso_features[model_key])
        X_scaled = self.lasso_scalers[model_key].transform(X_lasso)
        score = self.lasso_models[model_key].predict(X_scaled)[0]
        return {'status': status, 'score': score, 'model': model_key}

    def predict(self, X):
        """
        Prédit pour un ensemble d'étudiants.

        Args:
            X (pd.DataFrame): DataFrame contenant les données des étudiants

        Returns:
            pd.DataFrame: DataFrame original avec les colonnes de prédiction ajoutées
        """
        results_df = X.copy()
        predictions = []

        for _, student in X.iterrows():
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

        results_df['Prediction_Status'] = [p['status'] for p in predictions]
        results_df['Score_Predit'] = [p['score'] for p in predictions]
        results_df['Model_Utilise'] = [p['model'] for p in predictions]

        return results_df


def evaluate_predictions(df_result, rank_method='average'):
    """
    Évalue les prédictions du modèle.

    Args:
        df_result (pd.DataFrame): DataFrame contenant les prédictions
        rank_method (str): Méthode de ranking à utiliser

    Returns:
        tuple: (RMSE, MRR strict, MRR open)
    """
    df = df_result.copy()
    df['Rang_L1_New'] = df['Score L1'].rank(ascending=False, method=rank_method)
    df['Rang_Predit'] = df['Score_Predit'].rank(ascending=False, method=rank_method)
    df = df[['Rang_L1_New', 'Rang_Predit', 'RESULTAT']]

    rmse = np.sqrt(root_mean_squared_error(df['Rang_L1_New'], df['Rang_Predit']))

    df_strict = df.loc[df['RESULTAT'] == 'PASSE'].copy()
    df_open = df.loc[df['RESULTAT'] != 'NON ADMIS'].copy()

    df_strict.loc[:, 'mrr'] = 1 / df_strict['Rang_Predit']
    df_open.loc[:, 'mrr'] = 1 / df_open['Rang_Predit']

    return rmse, df_strict['mrr'].sum(), df_open['mrr'].sum()


def evaluate_all_predictions(df_results, suffix, rank_method='average'):
    """
    Évalue les prédictions sur plusieurs jeux de données.

    Args:
        df_results (list): Liste de DataFrames contenant les prédictions
        suffix (str): Suffixe pour l'affichage des résultats
        rank_method (str): Méthode de ranking à utiliser
    """
    print(f"\nEVALUATIONS POUR {suffix}")
    qualities = (0, 0, 0)

    for df_result in df_results:
        rmse, mrr_strict, mrr_open = evaluate_predictions(df_result, rank_method)
        qualities = np.add(qualities, (rmse, mrr_strict, mrr_open))

    qualities = qualities / len(df_results)

    print(f"RMSE_{suffix}_{rank_method} : {qualities[0]:.4f}")
    print(f"MRR_{suffix}_{rank_method} strict : {qualities[1]:.4f}")
    print(f"MRR_{suffix}_{rank_method} open : {qualities[2]:.4f}")


def evaluate(docs, predictors, suffix='', rank_method='average'):
    """
    Évalue le modèle sur plusieurs documents.

    Args:
        docs (list): Liste des DataFrames de test
        predictors (list): Liste des prédicteurs
        suffix (str): Suffixe pour l'affichage des résultats
        rank_method (str): Méthode de ranking à utiliser
    """
    results = []
    for i in range(len(docs)):
        results.append(predictors[i].predict(docs[i].copy()))

    evaluate_all_predictions(results, suffix, rank_method)


def preprocess_data(df):
    """
    Effectue le prétraitement des données sur le DataFrame donné.

    Args:
        df (pd.DataFrame): Le DataFrame à prétraiter.

    Returns:
        pd.DataFrame: Le DataFrame prétraité.
    """
    df = df.fillna(0)

    # Encodage des séries
    if 'Série' in df.columns:
        series = ['S1', 'S2', 'S3']
        for serie in series:
            df[serie] = df['Série'].apply(lambda x: 1 if x == serie else 0)
        df.drop('Série', axis=1, inplace=True)

    # Encodage du sexe
    if 'Sexe' in df.columns:
        df['Homme'] = df['Sexe'].apply(lambda x: 1 if x == 'M' else 0)
        df['Femme'] = df['Sexe'].apply(lambda x: 1 if x == 'F' else 0)
        df.drop('Sexe', axis=1, inplace=True)

    # Moyennes par Académie
    if "Académie de l'Ets. Prov." in df.columns and 'Moy. Gle' in df.columns:
        academie_mean = df.groupby("Académie de l'Ets. Prov.")['Moy. Gle'].mean().to_dict()
        df['Academie perf.'] = df["Académie de l'Ets. Prov."].map(academie_mean)
        df.drop("Académie de l'Ets. Prov.", axis=1, inplace=True)

    # Moyennes par Résidence
    if 'Résidence' in df.columns and 'Moy. Gle' in df.columns:
        residence_mean = df.groupby("Résidence")['Moy. Gle'].mean().to_dict()
        df['Residence perf.'] = df["Résidence"].map(residence_mean)
        df.drop("Résidence", axis=1, inplace=True)

    # Supprimer la colonne "Mention" si elle existe
    if 'Mention' in df.columns:
        df.drop('Mention', axis=1, inplace=True)

    return df


def load_data(filepath):
    """
    Charge et prétraite les données depuis un fichier CSV.

    Args:
        filepath (str): Chemin vers le fichier CSV

    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    df = pd.read_csv(filepath)
    return preprocess_data(df)

def predict_single_student(data_dict):
    """
    Prédit pour un seul étudiant à partir d'un dictionnaire de données.

    Args:
        data_dict (dict): Dictionnaire contenant les données d'un étudiant

    Returns:
        dict: Résultat de la prédiction contenant le statut et le score
    """
    # Créer un DataFrame d'une seule ligne à partir du dictionnaire
    student_df = pd.DataFrame([data_dict])

    # Initialiser le prédicteur
    predictor = DioresPredictorEnsemblisteLasso()

    # Faire la prédiction
    result = predictor.predict(student_df)

    # Retourner le résultat de la première (et seule) ligne
    return {
        'status': result['Prediction_Status'].iloc[0],
        'score': result['Score_Predit'].iloc[0],
        'model': result['Model_Utilise'].iloc[0]
    }

def calculate_success_probability(student_data):
    """
    Calcule le pourcentage de chance de réussite d'un élève.
    """
    prediction = predict_single_student(student_data)

    # # Normalisation du score prédit
    # score_max = 200  # Ajuster cette valeur selon l'échelle réelle de votre modèle
    # normalized_score = min(20, (prediction['score'] / score_max) * 20)
    # score_percentage = (normalized_score / 20) * 100

    # Normalisation du score prédit
    score_max = 120  # Ajuster cette valeur selon l'échelle réelle de votre modèle
    normalized_score = min(20, (prediction['score'] / score_max) * 20)
    score_percentage = (normalized_score / 20) * 100

    # Pondération basée sur le statut prédit
    status_weights = {
        'MENTION SUPÉRIEURE': 0.95,
        'PASSABLE': 0.75,
        'DEUXIÈME SESSION': 0.40,
        'NON ADMIS': 0.10
    }

    base_probability = status_weights.get(prediction['status'], 0)

    # Ajustements plus modérés
    adjustments = {
        'mention': {
            'Bien': 0.10,        # Réduit à +10%
            'Assez-Bien': 0.07,  # Réduit à +7%
            'Passable': 0.03     # Réduit à +3%
        },
        'serie': {
            'S1': 0.07,          # Réduit à +7%
            'S3': 0.07,          # Réduit à +7%
            'S2': 0.03           # Réduit à +3%
        }
    }

    # mention_boost = adjustments['mention'].get(student_data['Mention'], 0)
    # serie_boost = adjustments['serie'].get(student_data['Série'], 0)

    mention_boost = 0
    serie_boost = 0

    # Calcul de la probabilité finale avec plus de nuance
    raw_probability = base_probability + mention_boost + serie_boost

    # Ajustement basé sur le score normalisé
    score_factor = score_percentage / 100  # Entre 0 et 1
    final_probability = min(95, raw_probability * 100 * score_factor)  # Plafonnement à 95%

    return {
        'probabilité_globale': round(final_probability, 2),
        'détails': {
            'statut_prédit': prediction['status'],
            'score_prédit': round(normalized_score, 2),
            'score_pourcentage': round(score_percentage, 2),
            'facteur_statut': round(base_probability * 100, 2),
            'bonus_mention': round(mention_boost * 100, 2),
            'bonus_série': round(serie_boost * 100, 2)
        }
    }

def main():
    # """
    # Fonction principale exécutant l'évaluation du modèle.
    # """
    # # Obtenir le chemin absolu du répertoire contenant le script
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # app_dir = os.path.dirname(os.path.dirname(script_dir))
    # directory = os.path.join(app_dir, "app", "data", "Datasets", "L1MPI")
    #
    # # Charger les fichiers de test
    # doc1_df_test = load_data(os.path.join(directory, "doc1", "doc1_df_test.csv"))
    # doc2_df_test = load_data(os.path.join(directory, "doc2", "doc2_df_test.csv"))
    # doc3_df_test = load_data(os.path.join(directory, "doc3", "doc3_df_test.csv"))
    #
    # # Préparer les documents et les prédicteurs
    # docs = [doc1_df_test, doc2_df_test, doc3_df_test]
    # predictor = DioresPredictorEnsemblisteLasso()
    # predictors = [predictor] * 3
    #
    # # Évaluer le modèle
    # evaluate(docs, predictors, suffix='DIORES', rank_method='average')


    # # Exemple d'utilisation avec toutes les colonnes nécessaires
    # student_data = {
    #     # Informations personnelles
    #     'Sexe': 'M',
    #     'Série': 'S1',
    #     'Age en Décembre 2018': 19,
    #
    #     # Notes du BAC
    #     'MATH': 15.5,
    #     'SCPH': 14.0,
    #     'FR': 12.5,
    #     'PHILO': 11.0,
    #     'AN': 13.5,
    #
    #     # Moyennes
    #     'Moy. nde': 13.5,
    #     'Moy. ère': 14.0,
    #     'Moy. S Term.': 14.5,
    #     'Moy. S Term..1': 14.2,
    #     'Moy. Gle': 13.8,
    #     'Moy. sur Mat.Fond.': 14.7,
    #
    #     # Informations du BAC
    #     'Année BAC': 2018,
    #     'Nbre Fois au BAC': 1,
    #     'Mention': 'Assez-Bien',
    #
    #     # Points et résultats
    #     'Groupe Résultat': 1,
    #     'Tot. Pts au Grp.': 245,
    #     'Moyenne au Grp.': 14.2,
    #
    #     # Informations établissement et localisation
    #     'Résidence': 'Dakar',
    #     'Ets. de provenance': 'Lycée Seydina Limamou Laye',
    #     'Centre d\'Ec.': 'Dakar',
    #     'Académie de l\'Ets. Prov.': 'Dakar',
    #     'REGION_DE_NAISSANCE': 'Dakar'
    # }
    #
    # # Faire la prédiction
    # prediction = predict_single_student(student_data)
    #
    # # Afficher les résultats
    # print("\nRésultats de la prédiction:")
    # print("=" * 30)
    # print(f"Statut: {prediction['status']}")
    # print(f"Score prédit: {prediction['score']:.2f}")
    # print(f"Modèle utilisé: {prediction['model']}")

    students_data = [
        {
            # Étudiant 1 - Très bon élève de S1
            'Sexe': 'M',
            'Série': 'S1',
            'Age en Décembre 2018': 18,
            'MATH': 17.5,
            'SCPH': 16.0,
            'FR': 15.5,
            'PHILO': 14.0,
            'AN': 16.5,
            'Moy. nde': 16.0,
            'Moy. ère': 16.5,
            'Moy. S Term.': 16.8,
            'Moy. S Term..1': 16.5,
            'Moy. Gle': 16.3,
            'Moy. sur Mat.Fond.': 16.7,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 1,
            'Mention': 'Bien',
            'Groupe Résultat': 1,
            'Tot. Pts au Grp.': 320,
            'Moyenne au Grp.': 16.0,
            'Résidence': 'Dakar',
            'Ets. de provenance': 'Lycée Seydina Limamou Laye',
            'Centre d\'Ec.': 'Dakar',
            'Académie de l\'Ets. Prov.': 'Dakar',
            'REGION_DE_NAISSANCE': 'Dakar'
        },
        {
            # Étudiant 2 - Élève moyen de S2
            'Sexe': 'F',
            'Série': 'S2',
            'Age en Décembre 2018': 19,
            'MATH': 12.5,
            'SCPH': 13.0,
            'FR': 14.5,
            'PHILO': 13.0,
            'AN': 12.5,
            'Moy. nde': 12.8,
            'Moy. ère': 13.0,
            'Moy. S Term.': 13.2,
            'Moy. S Term..1': 13.0,
            'Moy. Gle': 13.1,
            'Moy. sur Mat.Fond.': 12.8,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 1,
            'Mention': 'Passable',
            'Groupe Résultat': 2,
            'Tot. Pts au Grp.': 260,
            'Moyenne au Grp.': 13.0,
            'Résidence': 'Thiès',
            'Ets. de provenance': 'Lycée Malick Sy',
            'Centre d\'Ec.': 'Thiès',
            'Académie de l\'Ets. Prov.': 'Thiès',
            'REGION_DE_NAISSANCE': 'Thiès'
        },
        {
            # Étudiant 3 - Élève en difficulté de S3
            'Sexe': 'M',
            'Série': 'S3',
            'Age en Décembre 2018': 20,
            'MATH': 10.5,
            'SCPH': 11.0,
            'FR': 12.0,
            'PHILO': 11.5,
            'AN': 11.0,
            'Moy. nde': 11.2,
            'Moy. ère': 11.0,
            'Moy. S Term.': 11.5,
            'Moy. S Term..1': 11.3,
            'Moy. Gle': 11.2,
            'Moy. sur Mat.Fond.': 10.8,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 2,
            'Mention': 'Passable',
            'Groupe Résultat': 3,
            'Tot. Pts au Grp.': 224,
            'Moyenne au Grp.': 11.2,
            'Résidence': 'Saint-Louis',
            'Ets. de provenance': 'Lycée Charles De Gaulle',
            'Centre d\'Ec.': 'Saint-Louis',
            'Académie de l\'Ets. Prov.': 'Saint-Louis',
            'REGION_DE_NAISSANCE': 'Saint-Louis'
        },
        {
            # Étudiant 4 - Très bonne élève de S2
            'Sexe': 'F',
            'Série': 'S2',
            'Age en Décembre 2018': 18,
            'MATH': 16.5,
            'SCPH': 17.0,
            'FR': 16.0,
            'PHILO': 15.5,
            'AN': 16.0,
            'Moy. nde': 16.2,
            'Moy. ère': 16.5,
            'Moy. S Term.': 16.3,
            'Moy. S Term..1': 16.4,
            'Moy. Gle': 16.3,
            'Moy. sur Mat.Fond.': 16.7,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 1,
            'Mention': 'Bien',
            'Groupe Résultat': 1,
            'Tot. Pts au Grp.': 326,
            'Moyenne au Grp.': 16.3,
            'Résidence': 'Dakar',
            'Ets. de provenance': 'Lycée Blaise Diagne',
            'Centre d\'Ec.': 'Dakar',
            'Académie de l\'Ets. Prov.': 'Dakar',
            'REGION_DE_NAISSANCE': 'Dakar'
        },
        {
            # Étudiant 5 - Élève moyen de S1
            'Sexe': 'M',
            'Série': 'S1',
            'Age en Décembre 2018': 19,
            'MATH': 13.5,
            'SCPH': 12.5,
            'FR': 13.0,
            'PHILO': 12.0,
            'AN': 13.5,
            'Moy. nde': 12.8,
            'Moy. ère': 13.2,
            'Moy. S Term.': 13.0,
            'Moy. S Term..1': 13.1,
            'Moy. Gle': 13.0,
            'Moy. sur Mat.Fond.': 13.0,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 1,
            'Mention': 'Passable',
            'Groupe Résultat': 2,
            'Tot. Pts au Grp.': 260,
            'Moyenne au Grp.': 13.0,
            'Résidence': 'Louga',
            'Ets. de provenance': 'Lycée Louga',
            'Centre d\'Ec.': 'Louga',
            'Académie de l\'Ets. Prov.': 'Louga',
            'REGION_DE_NAISSANCE': 'Louga'
        },
        {
            # Étudiant 6 - Bon élève de S3
            'Sexe': 'F',
            'Série': 'S3',
            'Age en Décembre 2018': 19,
            'MATH': 14.5,
            'SCPH': 15.0,
            'FR': 14.0,
            'PHILO': 13.5,
            'AN': 14.5,
            'Moy. nde': 14.3,
            'Moy. ère': 14.5,
            'Moy. S Term.': 14.2,
            'Moy. S Term..1': 14.4,
            'Moy. Gle': 14.3,
            'Moy. sur Mat.Fond.': 14.7,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 1,
            'Mention': 'Assez-Bien',
            'Groupe Résultat': 1,
            'Tot. Pts au Grp.': 286,
            'Moyenne au Grp.': 14.3,
            'Résidence': 'Kaolack',
            'Ets. de provenance': 'Lycée Kaolack',
            'Centre d\'Ec.': 'Kaolack',
            'Académie de l\'Ets. Prov.': 'Kaolack',
            'REGION_DE_NAISSANCE': 'Kaolack'
        },
        {
            # Étudiant 7 - Excellent élève de S1
            'Sexe': 'M',
            'Série': 'S1',
            'Age en Décembre 2018': 18,
            'MATH': 18.5,
            'SCPH': 17.5,
            'FR': 16.5,
            'PHILO': 16.0,
            'AN': 17.0,
            'Moy. nde': 17.2,
            'Moy. ère': 17.5,
            'Moy. S Term.': 17.3,
            'Moy. S Term..1': 17.4,
            'Moy. Gle': 17.3,
            'Moy. sur Mat.Fond.': 18.0,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 1,
            'Mention': 'Bien',
            'Groupe Résultat': 1,
            'Tot. Pts au Grp.': 346,
            'Moyenne au Grp.': 17.3,
            'Résidence': 'Dakar',
            'Ets. de provenance': 'Lycée Seydou Nourou Tall',
            'Centre d\'Ec.': 'Dakar',
            'Académie de l\'Ets. Prov.': 'Dakar',
            'REGION_DE_NAISSANCE': 'Dakar'
        },
        {
            # Étudiant 8 - Élève en difficulté de S2
            'Sexe': 'F',
            'Série': 'S2',
            'Age en Décembre 2018': 20,
            'MATH': 10.0,
            'SCPH': 11.5,
            'FR': 11.0,
            'PHILO': 10.5,
            'AN': 11.0,
            'Moy. nde': 10.8,
            'Moy. ère': 11.0,
            'Moy. S Term.': 10.9,
            'Moy. S Term..1': 11.1,
            'Moy. Gle': 10.9,
            'Moy. sur Mat.Fond.': 10.7,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 2,
            'Mention': 'Passable',
            'Groupe Résultat': 3,
            'Tot. Pts au Grp.': 218,
            'Moyenne au Grp.': 10.9,
            'Résidence': 'Ziguinchor',
            'Ets. de provenance': 'Lycée Djignabo',
            'Centre d\'Ec.': 'Ziguinchor',
            'Académie de l\'Ets. Prov.': 'Ziguinchor',
            'REGION_DE_NAISSANCE': 'Ziguinchor'
        },
        {
            # Étudiant 9 - Bon élève de S1
            'Sexe': 'M',
            'Série': 'S1',
            'Age en Décembre 2018': 19,
            'MATH': 15.5,
            'SCPH': 14.5,
            'FR': 15.0,
            'PHILO': 14.0,
            'AN': 15.0,
            'Moy. nde': 14.8,
            'Moy. ère': 15.0,
            'Moy. S Term.': 14.7,
            'Moy. S Term..1': 14.9,
            'Moy. Gle': 14.8,
            'Moy. sur Mat.Fond.': 15.0,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 1,
            'Mention': 'Assez-Bien',
            'Groupe Résultat': 1,
            'Tot. Pts au Grp.': 296,
            'Moyenne au Grp.': 14.8,
            'Résidence': 'Thiès',
            'Ets. de provenance': 'Lycée Malick Sy',
            'Centre d\'Ec.': 'Thiès',
            'Académie de l\'Ets. Prov.': 'Thiès',
            'REGION_DE_NAISSANCE': 'Thiès'
        },
        {
            # Étudiant 10 - Élève moyen de S3
            'Sexe': 'F',
            'Série': 'S3',
            'Age en Décembre 2018': 19,
            'MATH': 12.0,
            'SCPH': 13.0,
            'FR': 13.5,
            'PHILO': 12.5,
            'AN': 13.0,
            'Moy. nde': 12.8,
            'Moy. ère': 13.0,
            'Moy. S Term.': 12.9,
            'Moy. S Term..1': 13.1,
            'Moy. Gle': 12.9,
            'Moy. sur Mat.Fond.': 12.5,
            'Année BAC': 2018,
            'Nbre Fois au BAC': 1,
            'Mention': 'Passable',
            'Groupe Résultat': 2,
            'Tot. Pts au Grp.': 258,
            'Moyenne au Grp.': 12.9,
            'Résidence': 'Diourbel',
            'Ets. de provenance': 'Lycée Diourbel',
            'Centre d\'Ec.': 'Diourbel',
            'Académie de l\'Ets. Prov.': 'Diourbel',
            'REGION_DE_NAISSANCE': 'Diourbel'
        }
    ]

    # # Pour tester les prédictions pour tous les étudiants
    # for i, student_data in enumerate(students_data, 1):
    #     prediction = predict_single_student(student_data)
    #     print("\nRésultats de la prédiction:")
    #     print("=" * 30)
    #     print(f"Statut: {prediction['status']}")
    #     print(f"Score prédit: {prediction['score']:.2f}")
    #     print(f"Modèle utilisé: {prediction['model']}")

    # Exemple d'utilisation
    for i, student_data in enumerate(students_data, 1):
        probability = calculate_success_probability(student_data)
        print(f"\nÉtudiant {i}:")
        print("-" * 30)
        print(f"Profil: {student_data['Série']} - Moyenne: {student_data['Moy. Gle']}")
        print(f"Mention: {student_data['Mention']}")
        print("\nAnalyse des chances de réussite:")
        print(f"Probabilité globale: {probability['probabilité_globale']}%")
        print("\nDétails du calcul:")
        details = probability['détails']
        print(f"- Statut prédit: {details['statut_prédit']}")
        print(f"- Score prédit: {details['score_prédit']}/20 ({details['score_pourcentage']}%)")
        print(f"- Facteur lié au statut: {details['facteur_statut']}%")
        print(f"- Bonus mention: +{details['bonus_mention']}%")
        print(f"- Bonus série: +{details['bonus_série']}%")

if __name__ == "__main__":
    main()