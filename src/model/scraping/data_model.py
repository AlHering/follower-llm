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


def fix_schema(schema: str) -> str:
    """
    Function for fixing schema for populating infrastructure.
    :param schema: Schema string.
    :return: Fixed schema string.
    """
    schema = str(schema)
    if not schema.endswith("."):
        schema += "."
    return schema


def populate_data_instrastructure(engine: Engine, schema: str, model: dict) -> None:
    """
    Function for populating data infrastructure.
    :param engine: Database engine.
    :param schema: Schema for tables.
    :param model: Model dictionary for holding data classes.
    """
    schema = fix_schema(schema)
    base = declarative_base()

    class Source(base):
        """
        Source class, representing an source.
        """
        __tablename__ = f"{schema}source"
        __table_args__ = {
            "comment": "Source table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the source.")
        url = Column(String, unique=True,
                     comment="URL for the source.")
        name = Column(String,
                      comment="Display name for the source.")
        scraping_metadata = Column(JSON,
                                   comment="Metadata for scraping.")
        info = Column(JSON,
                      comment="Metadata of the source.")
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        feeds = relationship(
            "Feed", back_populates="source")
        channels = relationship(
            "Channel", back_populates="source")
        assets = relationship(
            "Asset", back_populates="source")

    class Feed(base):
        """
        Feed class, representing an source feed.
        """
        __tablename__ = f"{schema}feed"
        __table_args__ = {
            "comment": "Feed table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the feed.")
        url = Column(String, unique=True,
                     comment="URL for the feed.")
        name = Column(String,
                      comment="Display name for the feed.")
        scraping_metadata = Column(JSON,
                                   comment="Metadata for scraping.")
        info = Column(JSON,
                      comment="Metadata of the feed.")
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        source_id = mapped_column(
            Integer, ForeignKey(f"{schema}source.id"))
        source = relationship(
            "Source", back_populates="feeds")

    class Channel(base):
        """
        Channel class, representing an channel.
        """
        __tablename__ = f"{schema}channel"
        __table_args__ = {
            "comment": "Channel table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the channel.")
        url = Column(String, unique=True,
                     comment="URL for the channel.")
        name = Column(String,
                      comment="Display name for the channel.")
        scraping_metadata = Column(JSON,
                                   comment="Metadata for scraping.")
        info = Column(JSON,
                      comment="Metadata of the channel.")
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        source_id = mapped_column(
            Integer, ForeignKey(f"{schema}source.id"))
        source = relationship(
            "Source", back_populates="channels")
        assets = relationship(
            "Asset", back_populates="channel")

    class Asset(base):
        """
        Asset class, representing an asset.
        """
        __tablename__ = f"{schema}asset"
        __table_args__ = {
            "comment": "Asset table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the asset.")
        url = Column(String, unique=True,
                     comment="URL for the asset.")
        scraping_metadata = Column(JSON,
                                   comment="Metadata for scraping.")
        info = Column(JSON,
                      comment="Metadata of the asset.")
        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        files = relationship(
            "File", back_populates="asset")
        source_id = mapped_column(
            Integer, ForeignKey(f"{schema}source.id"))
        source = relationship(
            "Source", back_populates="assets")
        channel_id = mapped_column(
            Integer, ForeignKey(f"{schema}channel.id"))
        channel = relationship(
            "Channel", back_populates="assets")

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

    for dataclass in [Source, Channel, Feed, Asset, File]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)
