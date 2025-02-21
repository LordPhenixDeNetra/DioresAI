from pydantic import BaseModel
from typing import List, Optional


# Schemas pour Residence
class ResidenceBase(BaseModel):
    name: str


class ResidenceCreate(ResidenceBase):
    pass


class Residence(ResidenceBase):
    id: int
    academie_id: int

    class Config:
        from_attributes = True


# Schemas pour Academie
class AcademieBase(BaseModel):
    name: str


class AcademieCreate(AcademieBase):
    pass


class Academie(AcademieBase):
    id: int
    region_id: int
    residences: List[Residence] = []

    class Config:
        from_attributes = True


# Schemas pour Region
class RegionBase(BaseModel):
    name: str


class RegionCreate(RegionBase):
    pass


class Region(RegionBase):
    id: int
    academies: List[Academie] = []

    class Config:
        from_attributes = True
