import pandas as pd
from sklearn.preprocessing import LabelEncoder


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