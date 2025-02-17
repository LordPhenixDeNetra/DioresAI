import os
import pickle
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from dataframe_processor import DataFrameProcessor
import utils


class DioresPredictorEnsemblisteLasso(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.dt_paths = {
            'admission': './data/Models/V2/admi_non_admi_best_model_DecisionTree.pkl',
            'session': './data/Models/V2/session_best_model_DecisionTree.pkl',
            'mention': './data/Models/V2/mention_best_model_DecisionTree.pkl'

            # 'admission': '/content/drive/MyDrive/Memoire/DIORES/Models/V2/admi_non_admi_best_model_DecisionTree.pkl',
            # 'session': '/content/drive/MyDrive/Memoire/DIORES/Models/V2/session_best_model_DecisionTree.pkl',
            # 'mention': '/content/drive/MyDrive/Memoire/DIORES/Models/V2/mention_best_model_DecisionTree.pkl'
        }

        self.lasso_base_paths = {
            # 'non_admi': '/content/drive/MyDrive/Memoire/DIORES/Models/Lasso_Admi_Session/NON_ADMI/non_admi/',
            # 'deuxieme_session': '/content/drive/MyDrive/Memoire/DIORES/Models/Lasso_Admi_Session/DEUXIME_SESSION/deuxieme_session/',
            # 'passable': '/content/drive/MyDrive/Memoire/DIORES/Models/Lasso_Admi_Session/PASSABLE/passable/',
            # 'mention': '/content/drive/MyDrive/Memoire/DIORES/Models/Lasso_Admi_Session/MENTION/mention/'

            'non_admi': './data/Models/Lasso_Admi_Session/NON_ADMI/non_admi/',
            'deuxieme_session': './data/Models/Lasso_Admi_Session/DEUXIME_SESSION/deuxieme_session/',
            'passable': './data/Models/Lasso_Admi_Session/PASSABLE/passable/',
            'mention': './data/Models/Lasso_Admi_Session/MENTION/mention/'
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


directory = "./data/Datasets/L1MPI/"
print("Chemin du fichier :", os.path.abspath(os.path.join(directory, "doc1", "doc1_df_test.csv")))

# predictor = DioresPredictorEnsemblisteLasso()
# predictors = [predictor, predictor, predictor]
