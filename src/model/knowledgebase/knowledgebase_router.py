# -*- coding: utf-8 -*-
"""
****************************************************
*                    LLM Follower                  *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import traceback
from typing import Optional, Any, Union
from src.model.knowledgebase.abstract_knowledgebase import KnowledgeBase
from src.model.knowledgebase.chromadb_knowledgebase import ChromaKnowledgeBase

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
Interfacing
"""


def spawn_knowledgebase_instance(*args: Optional[Any], **kwargs: Optional[Any]) -> Union[KnowledgeBase, dict]:
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
        if kwargs.get("backend", args[0]) == "chromadb":
            return ChromaKnowledgeBase(peristant_directory=kwargs.get("knowledgebase_path", args[1]),
                                       metadata=kwargs.get(
                                           "knowledgebase_parameters", {}).get("metadata"),
                                       base_embedding_function=kwargs.get(
                                           "knowledgebase_parameters", {}).get("metadata"),
                                       implementation=kwargs.get("knowledgebase_parameters", {}).get("implementation"))
    except Exception as ex:
        return {"exception": ex, "trace": traceback.format_exc()}
