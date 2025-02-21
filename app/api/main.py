from fastapi import FastAPI, Depends, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.core.database import get_db, engine
from app.models import models
from app.schemas import schemas
from app.crud import crud
from typing import List

# from starlette.middleware.cors import CORSMiddleware


models.Base.metadata.create_all(bind=engine)

from app.api.models import StudentInput, PredictionResponse, PredictionResponseV2

from app.utils.all_files_V2 import predict_single_student, calculate_success_probability, \
    calculate_success_probability_V2

app = FastAPI(
    title="DIORES API",
    description="API pour la prédiction de la réussite des étudiants",
    version="1.0.0"
)

# Configuration correcte de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (évite les blocages)
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)


@app.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    try:
        # Tente une requête simple
        db.execute("SELECT 1")
        return {"message": "Connection à la base de données réussie!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de connexion à la base: {str(e)}")


@app.get("/check-tables")
def check_tables(db: Session = Depends(get_db)):
    # Obtient la liste de toutes les tables
    result = db.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in result]
    return {"tables": tables}


@app.get("/count-data")
def count_data(db: Session = Depends(get_db)):
    regions_count = db.query(models.Region).count()
    academies_count = db.query(models.Academie).count()
    residences_count = db.query(models.Residence).count()
    return {
        "regions": regions_count,
        "academies": academies_count,
        "residences": residences_count
    }


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


# Debug
def get_recommendation_V2(probability: float, probability_type: str) -> str:
    """Génère une recommandation basée sur la probabilité et le type de statut."""

    if probability_type == "orientation":
        if probability >= 75:
            return "Vous avez une très grande chance d'être orienté vers la filière."
        elif probability >= 50:
            return "Vous avez une grande chance d'être orienté vers la filière."
        elif probability >= 25:
            return "Vous avez une faible chance d'être orienté vers la filière."
        else:
            return "Vous avez une très faible chance d'être orienté vers la filière."

    elif probability_type == "success":
        if probability >= 95:
            return "Vous avez une très grande chance de réussir dans la filière."
        elif probability >= 70:
            return "Vous avez une grande chance de réussir dans la filière."
        elif probability >= 50:
            return "Vous avez une faible chance de réussir dans la filière."
        elif probability >= 10:
            return "Vous avez une très faible chance de réussir dans la filière."
        else:
            return "Vous avez une probabilité extrêmement faible de réussir dans la filière."

    else:
        return "Type de probabilité invalide. Veuillez utiliser 'orientation' ou 'success'."


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


@app.post("/predict_v2", response_model=PredictionResponseV2)
async def predict_student_v2(student: StudentInput):
    """
    Endpoint pour prédire le score L1, le rang et les probabilités de réussite et d'orientation.
    """
    try:
        # Préparation des données de l'étudiant
        student_data = format_student_data(student)

        # Calcul des probabilités et du score prédit
        # Faire la prédiction
        # prediction = predict_single_student(student_data)
        prediction = calculate_success_probability_V2(student_data)

        orientation_probability_message = get_recommendation_V2(prediction['orientation_probability'], "orientation")
        success_probability_message = get_recommendation_V2(prediction['success_probability'], "success")

        return {
            "status": prediction['statut_prédit'],
            "score": prediction['score_prédit'],
            "orientation_probability": prediction['orientation_probability'],
            "success_probability": prediction['success_probability'],
            "orientation_probability_message": orientation_probability_message,
            "success_probability_message": success_probability_message
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


@app.post("/regions/", response_model=schemas.Region)
def create_region(region: schemas.RegionCreate, db: Session = Depends(get_db)):
    return crud.create_region(db=db, region=region)


@app.get("/regions/", response_model=List[schemas.Region])
def read_regions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    regions = crud.get_regions(db, skip=skip, limit=limit)
    return regions


@app.post("/regions/{region_id}/academies/", response_model=schemas.Academie)
def create_academie_for_region(
        region_id: int, academie: schemas.AcademieCreate, db: Session = Depends(get_db)
):
    return crud.create_academie(db=db, academie=academie, region_id=region_id)


@app.post("/academies/{academie_id}/residences/", response_model=schemas.Residence)
def create_residence_for_academie(
        academie_id: int, residence: schemas.ResidenceCreate, db: Session = Depends(get_db)
):
    return crud.create_residence(db=db, residence=residence, academie_id=academie_id)


# Endpoints pour les Régions
@app.get("/regions/", response_model=List[schemas.Region], tags=["Régions"])
def list_regions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Liste toutes les régions"""
    regions = crud.get_regions(db, skip=skip, limit=limit)
    return regions


@app.get("/regions/{region_id}", response_model=schemas.Region, tags=["Régions"])
def get_region(region_id: int, db: Session = Depends(get_db)):
    """Obtient les détails d'une région spécifique"""
    region = crud.get_region(db, region_id=region_id)
    if not region:
        raise HTTPException(status_code=404, detail="Région non trouvée")
    return region


@app.get("/regions/{region_id}/details", tags=["Régions"])
def get_region_details(region_id: int, db: Session = Depends(get_db)):
    """Obtient les détails complets d'une région avec ses académies et résidences"""
    region = db.query(models.Region).filter(models.Region.id == region_id).first()
    if not region:
        raise HTTPException(status_code=404, detail="Région non trouvée")

    return {
        "region": region.name,
        "academies": [
            {
                "name": academie.name,
                "residences": [residence.name for residence in academie.residences]
            }
            for academie in region.academies
        ]
    }


# Endpoints pour les Académies
@app.get("/academies/", response_model=List[schemas.Academie], tags=["Académies"])
def list_academies(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Liste toutes les académies"""
    academies = crud.get_academies(db, skip=skip, limit=limit)
    return academies


@app.get("/academies/{academie_id}", response_model=schemas.Academie, tags=["Académies"])
def get_academie(academie_id: int, db: Session = Depends(get_db)):
    """Obtient les détails d'une académie spécifique"""
    academie = crud.get_academie(db, academie_id=academie_id)
    if not academie:
        raise HTTPException(status_code=404, detail="Académie non trouvée")
    return academie


@app.get("/regions/{region_id}/academies", response_model=List[schemas.Academie], tags=["Académies"])
def get_academies_by_region(region_id: int, db: Session = Depends(get_db)):
    """Liste toutes les académies d'une région spécifique"""
    region = crud.get_region(db, region_id=region_id)
    if not region:
        raise HTTPException(status_code=404, detail="Région non trouvée")
    return region.academies


# Endpoints pour les Résidences
@app.get("/residences/", response_model=List[schemas.Residence], tags=["Résidences"])
def list_residences(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Liste toutes les résidences"""
    residences = crud.get_residences(db, skip=skip, limit=limit)
    return residences


@app.get("/residences/{residence_id}", response_model=schemas.Residence, tags=["Résidences"])
def get_residence(residence_id: int, db: Session = Depends(get_db)):
    """Obtient les détails d'une résidence spécifique"""
    residence = crud.get_residence(db, residence_id=residence_id)
    if not residence:
        raise HTTPException(status_code=404, detail="Résidence non trouvée")
    return residence


@app.get("/academies/{academie_id}/residences", response_model=List[schemas.Residence], tags=["Résidences"])
def get_residences_by_academie(academie_id: int, db: Session = Depends(get_db)):
    """Liste toutes les résidences d'une académie spécifique"""
    academie = crud.get_academie(db, academie_id=academie_id)
    if not academie:
        raise HTTPException(status_code=404, detail="Académie non trouvée")
    return academie.residences


# Endpoint pour les statistiques
@app.get("/stats", tags=["Statuesque"])
def get_stats(db: Session = Depends(get_db)):
    """Obtient des statistiques générales sur les données"""
    return {
        "total_regions": db.query(models.Region).count(),
        "total_academies": db.query(models.Academie).count(),
        "total_residences": db.query(models.Residence).count()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
