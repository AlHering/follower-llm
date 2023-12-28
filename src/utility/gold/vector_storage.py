# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import traceback
from pydantic import BaseModel
from typing import List, Tuple, Any, Callable, Optional, Type, Union
from uuid import uuid4
from datetime import datetime as dt


"""
Vector DB backend overview
------------------------------------------
ChromaDB
 
SQLite-VSS
 
FAISS

PGVector
 
Qdrant
 
Pinecone

Redis

Langchain Vector DB Zoo
"""


"""
Vector store instantiation functions
"""


"""
Abstractions
"""


"""
Interfacing
"""


def spawn_knowledgebase_instance(*args: Optional[Any], **kwargs: Optional[Any]) -> Union[Any, dict]:
    """
    Function for spawning knowledgebase instances based on configuration arguments.
    :param args: Arbitrary initiation arguments.
    :param kwargs: Arbitrary initiation keyword arguments.
    :return: Language model instance if configuration was successful else an error report.
    """
    # TODO: Research common parameter pattern for popular knowledgebase backends
    # TODO: Update interfacing and move to gold utility
    # TODO: Support ChromaDB, SQLite-VSS, FAISS, PGVector, Qdrant, Pinecone, Redis, Langchain Vector DB Zoo(?)
    try:
        pass
    except Exception as ex:
        return {"exception": ex, "trace": traceback.format_exc()}
