# -*- coding: utf-8 -*-
"""
****************************************************
*                 General LLM App                  *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, List
from langchain.docstore.document import Document
from src.utility.bronze.hashing_utility import hash_text_with_sha256
from src.model.knowledgebase.abstract_knowledgebase import KnowledgeBase
from src.utility.bronze import json_utility


class OntologyInferringKnowledgebase(KnowledgeBase):
    """
    Class for handling knowledge base interaction with ChromaDB, optimized by ontology inference.
    """

    def __init__(self, inference_llm: Any, core_knowledgebase: KnowledgeBase, config_path: str) -> None:
        """
        Initiation method.
        :param inference_llm: LLM instance for inference.
        :param core_knowledgebase: Core knowledgebase.
        :param config_path: Config path.
        """
        self.inference_llm = inference_llm
        self.config_path = config_path
        if os.path.exists(self.config_path):
            self.config = json_utility.load(self.config_path)
        else:
            self.config = {
                "categories": {}
            }
        self.set_core_knowledgebase(core_knowledgebase)

    def set_core_knowledgebase(self, core_knowledgebase: KnowledgeBase) -> None:
        """
        Method for routing knowledgebase methods to core knowledgebase.
        :param core_knowledgebase: Core knowledgebase.
        """
        self.core_knowlegebase = core_knowledgebase
        self.delete_document = core_knowledgebase.delete_document
        self.wipe_knowledgebase = core_knowledgebase.wipe_knowledgebase
        self.get_or_create_collection = core_knowledgebase.get_or_create_collection
        self.get_retriever = core_knowledgebase.get_retriever
        self.embed_documents = core_knowledgebase.embed_documents
        self.retrieve_documents = core_knowledgebase.retrieve_documents
        self.get_retriever = core_knowledgebase.get_retriever

    # Override
    def embed_documents(self, documents: List[Document], metadatas: List[dict] = None, ids: List[str] = None, collection: str = "base", compute_additional_metadata: bool = False) -> None:
        """
        Method for embedding documents.
        :param documents: Documents to embed.
        :param metadatas: Metadata entries. 
            Defaults to None.
        :param ids: Custom IDs to add. 
            Defaults to the hash of the document contents.
        :param collection: Collection to use.
            Defaults to "base".
        :param compute_additional_metadata: Flag for declaring, whether to compute additional metadata.
            Defaults to False.
        """
        if metadatas is None:
            metadatas = [{} for _ in documents]
        if compute_additional_metadata:
            for doc_index, doc_content in enumerate(documents):
                metadatas[doc_index] = metadatas[doc_index].update(
                    self.compute_additional_metadata(doc_content, collection))

        self.collections[collection].add_documents(documents=documents, metadatas=metadatas, ids=[
            hash_text_with_sha256(document.page_content) for document in documents] if ids is None else ids)

        self.collections[collection].persist()
