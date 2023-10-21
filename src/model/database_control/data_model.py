# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from sqlalchemy.orm import relationship, mapped_column, declarative_base
from sqlalchemy import Engine, Column, String, JSON, ForeignKey, Integer, DateTime, func, Uuid, Text, event, Boolean
from uuid import uuid4, UUID
from typing import Any


def populate_data_instrastructure(engine: Engine, schema: str, model: dict) -> None:
    """
    Function for populating data infrastructure.
    :param engine: Database engine.
    :param schema: Schema for tables.
    :param model: Model dictionary for holding data classes.
    """
    schema = str(schema)
    if not schema.endswith("."):
        schema += "."
    base = declarative_base()

    class User(base):
        """
        User class, representing an user.
        """
        __tablename__ = f"{schema}user"
        __table_args__ = {
            "comment": "User table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the User.")
        email = Column(String, nullable=False, unique=True,
                       comment="User email.")
        password_hash = Column(String, nullable=False
                       comment="User password hash.")
        config = Column(JSON, nullable=False,
                        comment="User configuration.")
        
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        assets = relationship(
            "Asset", back_populates="owner")
        configs = relationship(
            "Config", back_populates="owner")
        granted = relationship(
            "Access", back_populates="granter")

    class Asset(base):
        """
        Asset class, representing an asset.
        """
        __tablename__ = f"{schema}asset"
        __table_args__ = {
            "comment": "Asset table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the asset.")
        url = Column(String,
                     comment="URL for the asset.")
        source = Column(String,
                        comment="Main metadata source for the asset.")
        sha256 = Column(Text,
                        comment="SHA256 hash for the asset.")
        meta_data = Column(JSON,
                           comment="Metadata of the asset.")
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        files = relationship(
            "File", back_populates="asset")
        owner_id = mapped_column(
            Integer, ForeignKey(f"{schema}user.id"))
        owner = relationship(
            "User", back_populates="assets")

    class File(base):
        """
        File class, representing a file.
        """
        __tablename__ = f"{schema}document"
        __table_args__ = {
            "comment": "File table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the file.")
        path = Column(String, nullable=False,
                      comment="Path of the file.")
        encoding = Column(String, nullable=False,
                      comment="Encoding of the file.")
        extension = Column(String, nullable=False,
                      comment="Extension of the file.")
        url = Column(String,
                     comment="URL for the file.")
        source = Column(String,
                        comment="Main metadata source for the file.")
        sha256 = Column(Text,
                        comment="SHA256 hash for the file.")
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        asset_id = mapped_column(
            Integer, ForeignKey(f"{schema}asset.id"))
        asset = relationship(
            "Asset", back_populates="files")
        
    class Config(base):
        """
        Config class, representing a config.
        """
        __tablename__ = f"{schema}config"
        __table_args__ = {
            "comment": "Config table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the config.")
        type = Column(String, nullable=False,
                      comment="Target object type of the file.")
        config = Column(JSON, nullable=False,
                        
                        comment="Object configuration.")
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        owner_id = mapped_column(
            Integer, ForeignKey(f"{schema}user.id"))
        owner = relationship(
            "User", back_populates="configs")
        
    class Access(base):
        """
        Access class, representing access rights.
        """
        __tablename__ = f"{schema}config"
        __table_args__ = {
            "comment": "Access table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the access.")
        type = Column(String, nullable=False,
                      comment="Target object type.")
        target_id = Column(Integer, nullable=False,
                    comment="ID of the target object.")
        user_id = Column(Integer, nullable=False,
                    comment="ID of the user.")
        level = Column(Integer, nullable=False, default=0,
                    comment="Access level.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        granter_id = mapped_column(
            Integer, ForeignKey(f"{schema}user.id"))
        granter = relationship(
            "User", back_populates="granted")

    class Log(base):
        """
        Log class, representing an log entry.
        """
        __tablename__ = f"{schema}log"
        __table_args__ = {
            "comment": "Log table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, autoincrement=True, unique=True, nullable=False,
                    comment="ID of the logging entry.")
        request = Column(JSON, nullable=False,
                         comment="Request, sent to the backend.")
        response = Column(JSON, comment="Response, given by the backend.")
        requested = Column(DateTime, server_default=func.now(),
                           comment="Timestamp of request recieval.")
        responded = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                           comment="Timestamp of reponse transmission.")

    for dataclass in [User, Asset, File, Config, Access, Log]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)
