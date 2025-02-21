from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Region(Base):
    __tablename__ = "regions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

    # Relation one-to-many avec Academie
    academies = relationship("Academie", back_populates="region")


class Academie(Base):
    __tablename__ = "academies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    region_id = Column(Integer, ForeignKey("regions.id"))

    # Relations
    region = relationship("Region", back_populates="academies")
    residences = relationship("Residence", back_populates="academie")


class Residence(Base):
    __tablename__ = "residences"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    academie_id = Column(Integer, ForeignKey("academies.id"))

    # Relation many-to-one avec Academie
    academie = relationship("Academie", back_populates="residences")
