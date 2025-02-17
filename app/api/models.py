from pydantic import BaseModel
from typing import Dict, Optional, List


# Modèles Pydantic pour la validation des données
class StudentInput(BaseModel):
    Sexe: str
    Série: str
    Age_en_Décembre_2018: int
    MATH: float
    SCPH: float
    FR: float
    PHILO: float
    AN: float
    Moy_nde: float
    Moy_ère: float
    Moy_S_Term: float
    Moy_S_Term_1: float
    Moy_Gle: float
    Moy_sur_Mat_Fond: float
    Année_BAC: int
    Nbre_Fois_au_BAC: int
    Mention: str
    Groupe_Résultat: int
    Tot_Pts_au_Grp: int
    Moyenne_au_Grp: float
    Résidence: str
    Ets_de_provenance: str
    Centre_Ec: str
    Académie_de_Ets_Prov: str
    REGION_DE_NAISSANCE: str

    class Config:
        schema_extra = {
            "example": {
                "Sexe": "M",
                "Série": "S1",
                "Age_en_Décembre_2018": 18,
                "MATH": 17.5,
                "SCPH": 16.0,
                "FR": 15.5,
                "PHILO": 14.0,
                "AN": 16.5,
                "Moy_nde": 16.0,
                "Moy_ère": 16.5,
                "Moy_S_Term": 16.8,
                "Moy_S_Term_1": 16.5,
                "Moy_Gle": 16.3,
                "Moy_sur_Mat_Fond": 16.7,
                "Année_BAC": 2018,
                "Nbre_Fois_au_BAC": 1,
                "Mention": "Bien",
                "Groupe_Résultat": 1,
                "Tot_Pts_au_Grp": 320,
                "Moyenne_au_Grp": 16.0,
                "Résidence": "Dakar",
                "Ets_de_provenance": "Lycée Seydina Limamou Laye",
                "Centre_Ec": "Dakar",
                "Académie_de_Ets_Prov": "Dakar",
                "REGION_DE_NAISSANCE": "Dakar"
            }
        }


class PredictionResponse(BaseModel):
    status: str
    score: float
    probability: float
    details: Dict[str, float]
    recommendation: str