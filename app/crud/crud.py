from sqlalchemy.orm import Session
from typing import List, Optional
from ..models.models import Region, Academie, Residence
from ..schemas import schemas


# CRUD pour Region
def create_region(db: Session, region: schemas.RegionCreate) -> Region:
    db_region = Region(name=region.name)
    db.add(db_region)
    db.commit()
    db.refresh(db_region)
    return db_region


def get_region(db: Session, region_id: int) -> Optional[Region]:
    return db.query(Region).filter(Region.id == region_id).first()


def get_regions(db: Session, skip: int = 0, limit: int = 100) -> List[Region]:
    return db.query(Region).offset(skip).limit(limit).all()


# CRUD pour Academie
def create_academie(db: Session, academie: schemas.AcademieCreate, region_id: int) -> Academie:
    db_academie = Academie(**academie.dict(), region_id=region_id)
    db.add(db_academie)
    db.commit()
    db.refresh(db_academie)
    return db_academie


def get_academie(db: Session, academie_id: int) -> Optional[Academie]:
    return db.query(Academie).filter(Academie.id == academie_id).first()


def get_academies(db: Session, skip: int = 0, limit: int = 100) -> List[Academie]:
    return db.query(Academie).offset(skip).limit(limit).all()


# CRUD pour Residence
def create_residence(db: Session, residence: schemas.ResidenceCreate, academie_id: int) -> Residence:
    db_residence = Residence(**residence.dict(), academie_id=academie_id)
    db.add(db_residence)
    db.commit()
    db.refresh(db_residence)
    return db_residence


def get_residence(db: Session, residence_id: int) -> Optional[Residence]:
    return db.query(Residence).filter(Residence.id == residence_id).first()


def get_residences(db: Session, skip: int = 0, limit: int = 100) -> List[Residence]:
    return db.query(Residence).offset(skip).limit(limit).all()
