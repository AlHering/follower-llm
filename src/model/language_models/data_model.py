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
        password_hash = Column(String, nullable=False,
                               comment="User password hash.")
        config = Column(JSON, nullable=False,
                        comment="User configuration.")

        created = Column(DateTime, server_default=func.now(),
                         comment="Timestamp of creation.")
        updated = Column(DateTime, server_default=func.now(), server_onupdate=func.now(),
                         comment="Timestamp of last update.")
        inactive = Column(Boolean, nullable=False, default=False,
                          comment="Inactivity flag.")

        lm_instances = relationship(
            "LMInstance", back_populates="owner")
        kb_instances = relationship(
            "KBInstance", back_populates="owner")
        granted = relationship(
            "Access", back_populates="granter")

    class LMInstance(base):
        """
        Config class, representing a LM instance.
        """
        __tablename__ = f"{schema}lminstance"
        __table_args__ = {
            "comment": "LM instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the model instance.")
        backend = Column(String, nullable=False,
                         comment="Backend of the model instance.")
        model_path = Column(String, nullable=False,
                            comment="Path of the model instance.")

        model_file = Column(String,
                            comment="File of the model instance.")
        model_parameters = Column(JSON,
                                  comment="Parameters for the model instantiation.")
        tokenizer_path = Column(String,
                                comment="Path of the tokenizer.")
        tokenizer_parameters = Column(JSON,
                                      comment="Parameters for the tokenizer instantiation.")
        config_path = Column(String,
                             comment="Path of the config.")
        config_parameters = Column(JSON,
                                   comment="Parameters for the config.")

        default_system_prompt = Column(String,
                                       comment="Default system prompt of the model instance.")
        use_history = Column(Boolean, default=True,
                             comment="Flag for declaring whether to use a history.")
        encoding_parameters = Column(JSON,
                                     comment="Parameters for prompt encoding.")
        generating_parameters = Column(JSON,
                                       comment="Parameters for the response generation.")
        decoding_parameters = Column(JSON,
                                     comment="Parameters for response decoding.")

        resource_requirements = Column(JSON,
                                       comment="Resource profile for validating requirements.")

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

    class KBInstance(base):
        """
        Config class, representing a KB instance.
        """
        __tablename__ = f"{schema}kbinstance"
        __table_args__ = {
            "comment": "KB instance table.", "extend_existing": True}

        id = Column(Integer, primary_key=True, unique=True, nullable=False, autoincrement=True,
                    comment="ID of the knowledgebase instance.")
        backend = Column(String, nullable=False,
                         comment="Backend of the knowledgebase instance.")
        knowledgebase_path = Column(String, nullable=False, unique=True,
                                    comment="Path of the knowledgebase instance.")
        knowledgebase_parameters = Column(JSON,
                                          comment="Parameters for the knowledgebase instantiation.")
        preprocessing_parameters = Column(JSON,
                                          comment="Parameters for document preprocessing.")
        embedding_parameters = Column(JSON,
                                      comment="Parameters for document embedding.")
        retrieval_parameters = Column(JSON,
                                      comment="Parameters for the document retrieval.")

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
        embedding_model_instance_id = mapped_column(
            Integer, ForeignKey(f"{schema}lminstance.id"))
        embedding_model_instance = relationship(
            "LMInstance")

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

    for dataclass in [User, LMInstance, KBInstance, Access, Log]:
        model[dataclass.__tablename__.replace(schema, "")] = dataclass

    base.metadata.create_all(bind=engine)
