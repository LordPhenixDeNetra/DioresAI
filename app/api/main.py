from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.cors import CORSMiddleware


from app.api.models import StudentInput, PredictionResponse
from app.utils.all_files_V2 import predict_single_student, calculate_success_probability

app = FastAPI(
    title="DIORES API",
    description="API pour la prédiction de la réussite des étudiants",
    version="1.0.0"
)

## Configuration CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8080", "http://192.168.1.10:8080"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Configuration correcte de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (évite les blocages)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)

def format_student_data(student_input: StudentInput) -> dict:
    """Convertit les données d'entrée au format attendu par le modèle"""
    return {
        'Sexe': student_input.Sexe,
        'Série': student_input.Série,
        'Age en Décembre 2018': student_input.Age_en_Décembre_2018,
        'MATH': student_input.MATH,
        'SCPH': student_input.SCPH,
        'FR': student_input.FR,
        'PHILO': student_input.PHILO,
        'AN': student_input.AN,
        'Moy. nde': student_input.Moy_nde,
        'Moy. ère': student_input.Moy_ère,
        'Moy. S Term.': student_input.Moy_S_Term,
        'Moy. S Term..1': student_input.Moy_S_Term_1,
        'Moy. Gle': student_input.Moy_Gle,
        'Moy. sur Mat.Fond.': student_input.Moy_sur_Mat_Fond,
        'Année BAC': student_input.Année_BAC,
        'Nbre Fois au BAC': student_input.Nbre_Fois_au_BAC,
        'Mention': student_input.Mention,
        'Groupe Résultat': student_input.Groupe_Résultat,
        'Tot. Pts au Grp.': student_input.Tot_Pts_au_Grp,
        'Moyenne au Grp.': student_input.Moyenne_au_Grp,
        'Résidence': student_input.Résidence,
        'Ets. de provenance': student_input.Ets_de_provenance,
        'Centre d\'Ec.': student_input.Centre_Ec,
        'Académie de l\'Ets. Prov.': student_input.Académie_de_Ets_Prov,
        'REGION_DE_NAISSANCE': student_input.REGION_DE_NAISSANCE
    }

def get_recommendation(probability: float, status: str) -> str:
    """Génère une recommandation basée sur la probabilité et le statut"""
    if probability >= 75:
        return "Très grande chance de réussite. Votre profil est adéquat pour la formation."
    elif probability >= 50:
        return "Grande chance de réussite. Votre profil est adéquat pour la formation."
    elif probability >= 25:
        return "Faible chance de réussite. Votre profil peut ne pas être adéquat pour la formation."
    else:
        return "Très faible chance de réussite. Votre profil n'est probablement pas adéquat pour la formation."


@app.post("/predict", response_model=PredictionResponse)
async def predict_student(student: StudentInput):
    try:
        # Formater les données
        student_data = format_student_data(student)

        # Faire la prédiction
        prediction = predict_single_student(student_data)
        probability = calculate_success_probability(student_data)

        # Générer la recommandation
        recommendation = get_recommendation(
            probability['probabilité_globale'],
            prediction['status']
        )

        return {
            "status": prediction['status'],
            "score": prediction['score'],
            "probability": probability['probabilité_globale'],
            "details": {
                "score_percentage": probability['détails']['score_pourcentage'],
                "base_probability": probability['détails']['facteur_statut'],
                "mention_bonus": probability['détails']['bonus_mention'],
                "serie_bonus": probability['détails']['bonus_série']
            },
            "recommendation": recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_students(students: List[StudentInput]):
    try:
        results = []
        for student in students:
            student_data = format_student_data(student)
            prediction = predict_single_student(student_data)
            probability = calculate_success_probability(student_data)

            results.append({
                "student": student.dict(),
                "prediction": {
                    "status": prediction['status'],
                    "score": prediction['score'],
                    "probability": probability['probabilité_globale'],
                    "recommendation": get_recommendation(
                        probability['probabilité_globale'],
                        prediction['status']
                    )
                }
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)