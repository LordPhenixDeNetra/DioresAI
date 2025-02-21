import json
from sqlalchemy.orm import Session

from app.core.database import Base, engine, SessionLocal
from app.models.models import Residence, Region, Academie


# from models.models import Region, Academie, Residence
# from core.database import SessionLocal, engine, Base


def init_db(db: Session, json_file: str):
    # Charger les données JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        # Pour chaque région dans le JSON
        for region_name, academies_data in data.items():
            # Créer la région
            region = Region(name=region_name)
            db.add(region)
            db.flush()  # Pour obtenir l'ID de la région

            # Pour chaque académie dans la région
            for academie_name, residences_list in academies_data.items():
                # Créer l'académie
                academie = Academie(
                    name=academie_name,
                    region_id=region.id
                )
                db.add(academie)
                db.flush()  # Pour obtenir l'ID de l'académie

                # Pour chaque résidence dans l'académie
                for residence_name in residences_list:
                    # Créer la résidence
                    residence = Residence(
                        name=residence_name,
                        academie_id=academie.id
                    )
                    db.add(residence)

        # Commit toutes les modifications
        db.commit()
        print("Base de données initialisée avec succès!")

    except Exception as e:
        db.rollback()
        print(f"Erreur lors de l'initialisation de la base de données: {str(e)}")
        raise


def main():
    # Recréer les tables
    Base.metadata.drop_all(bind=engine)  # Supprime toutes les tables existantes
    Base.metadata.create_all(bind=engine)  # Recrée toutes les tables

    # Créer une session de base de données
    db = SessionLocal()
    try:
        # Initialiser la base de données avec les données du fichier JSON
        init_db(db, 'mapping_region_academie_residence.json')
    finally:
        db.close()


if __name__ == "__main__":
    main()
