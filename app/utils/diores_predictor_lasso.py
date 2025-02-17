import os
import pickle


def load_model_and_scaler(model_path):
    """
    Charge un modèle et un scaler à partir du chemin donné.

    Args:
        model_path (str): Chemin du dossier contenant le modèle et le scaler.

    Returns:
        tuple: Le modèle et le scaler chargés.
    """
    with open(os.path.join(model_path, 'lasso_globale_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_path, 'lasso_globale_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

directory = "/data/Models/LassoSimple/"
# directory = "/content/drive/MyDrive/Memoire/DIORES/Models/LassoSimple/"

predictors = []
for i in range(1, 4):
    model_path = os.path.join(directory, f"Doc{i}")
    predictor, scaler = load_model_and_scaler(model_path)
    predictors.append(predictor)


class DioresPredictorLasso:
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.features = features

    def predict(self, df):
        """
        Fait des prédictions sur un DataFrame et renvoie les résultats au format attendu.
        """
        # Créer une copie du DataFrame pour ne pas modifier l'original
        df_result = df.copy()

        # Vérifier les colonnes manquantes
        missing_cols = set(self.features) - set(df.columns)
        if missing_cols:
            print(f"Attention: Colonnes manquantes: {missing_cols}")

            raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing_cols}")

        # Sélectionner et standardiser les features
        try:
            X = df_result[list(self.features)]
            X_scaled = self.scaler.transform(X)

            # Faire les prédictions
            predictions = self.model.predict(X_scaled)

            # Ajouter les prédictions au DataFrame résultat
            df_result['Score_Predit'] = predictions

            return df_result
        except Exception as e:
            print(f"Erreur lors de la prédiction: {str(e)}")
            print(f"Features attendues: {self.features}")
            print(f"Colonnes disponibles: {df_result.columns.tolist()}")
            raise

# Charger les modèles et créer les prédicteurs
# directory = "/content/drive/MyDrive/Memoire/DIORES/Models/LassoSimple/"
directory = "/data/Models/LassoSimple/"
predictors = []

for i in range(1, 4):
    model_path = os.path.join(directory, f"Doc{i}")
    # Charger le modèle et le scaler
    model, scaler = load_model_and_scaler(model_path)

    # Charger les informations sur les features
    with open(os.path.join(model_path, 'lasso_globale_info.pkl'), 'rb') as f:
        info = pickle.load(f)

    # Créer le prédicteur avec les features spécifiques
    predictor = DioresPredictorLasso(model, scaler, info['features'])
    predictors.append(predictor)