# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from time import sleep
from datetime import datetime as dt
from typing import Optional, Any, List, Dict, Union
from src.configuration import configuration as cfg
from src.utility.gold.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask as FilterMask
from src.utility.bronze import sqlalchemy_utility
from src.utility.bronze.hashing_utility import hash_text_with_sha256
from src.model.database.data_model import populate_data_instrastructure
from src.model.language_models.llm_pool import ThreadedLLMPool
from langchain.chains import RetrievalQA
from src.utility.silver import embedding_utility
from src.utility.bronze.hashing_utility import hash_text_with_sha256
from src.utility.silver.file_system_utility import safely_create_path
from src.model.knowledgebase.chromadb_knowledgebase import ChromaKnowledgeBase, KnowledgeBase


class FollowerLLMController(BasicSQLAlchemyInterface):
    """
    Controller class for handling backend interface requests.
    """

    def __init__(self, working_directory: str = None, database_uri: str = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
            Defaults to folder 'processes' folder under standard backend data path.
        :param database_uri: Database URI.
            Defaults to 'backend.db' file under default data path.
        """
        # Main instance variables
        self._logger = cfg.LOGGER
        self.working_directory = cfg.PATHS.BACKEND_PATH if working_directory is None else working_directory
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.database_uri = f"sqlite:///{os.path.join(cfg.PATHS.DATA_PATH, 'backend.db')}" if database_uri is None else database_uri

        # Database infrastructure
        super().__init__(self.working_directory, self.database_uri,
                         populate_data_instrastructure, self._logger)
        self.base = None
        self.engine = None
        self.model = None
        self.schema = None
        self.session_factory = None
        self.primary_keys = None
        self._setup_database()

        # Knowledgebase infrastructure
        self.knowledgebase_directory = os.path.join(
            self.working_directory, "knowledgebases")
        self.document_directory = os.path.join(
            self.working_directory, "library")
        safely_create_path(self.knowledgebase_directory)
        safely_create_path(self.document_directory)
        self.default_embedding_function = embedding_utility.LocalHuggingFaceEmbeddings(
            cfg.PATHS.INSTRUCT_XL_PATH
        )
        self.kbs: Dict[str, KnowledgeBase] = {}
        self.documents = {}
        for kb in self.get_objects_by_type("knowledgebase"):
            self.register_knowledgebase(kb.id, kb.handler, kb.persinstant_directory,
                                        kb.meta_data, kb.embedding_instance_id, kb.implementation)

        # LLM infrastructure
        self.llm_pool = ThreadedLLMPool()

        # Cache
        self._cache = {
            "active": {}
        }

    """
    Setup and population methods
    """

    def _setup_database(self) -> None:
        """
        Internal method for setting up database infastructure.
        """
        self._logger.info("Automapping existing structures")
        self.base = sqlalchemy_utility.automap_base()
        self.engine = sqlalchemy_utility.get_engine(
            f"sqlite:///{os.path.join(cfg.PATHS.DATA_PATH, 'backend.db')}" if self.database_uri is None else self.database_uri)

        self.model = {}
        self.schema = "backend."

        self._logger.info(
            f"Generating model tables for website with schema {self.schema}")
        populate_data_instrastructure(
            self.engine, self.schema, self.model)

        self.base.prepare(autoload_with=self.engine)
        self.session_factory = sqlalchemy_utility.get_session_factory(
            self.engine)
        self._logger.info("base created with")
        self._logger.info(f"Classes: {self.base.classes.keys()}")
        self._logger.info(f"Tables: {self.base.metadata.tables.keys()}")

        self.primary_keys = {
            object_class: self.model[object_class].__mapper__.primary_key[0].name for object_class in self.model}
        self._logger.info(f"Datamodel after addition: {self.model}")
        for object_class in self.model:
            self._logger.info(
                f"Object type '{object_class}' currently has {self.get_object_count_by_type(object_class)} registered entries.")

    """
    Exit and shutdown methods
    """

    def shutdown(self) -> None:
        """
        Method for running shutdown process.
        """
        self.llm_pool.stop_all()
        while any(self.llm_pool.is_running(instance_id) for instance_id in self._cache):
            sleep(2.0)

    """
    LLM handling methods
    """

    def load_instance(self, config_id: Union[str, int]) -> Optional[str]:
        """
        Method for loading a configured language model instance.
        :param config_id: Config ID.
        :return: Config ID if process as successful.
        """
        config_id = str(config_id)
        if config_id in self._cache:
            if not self.llm_pool.is_running(config_id):
                self.llm_pool.start(config_id)
                self._cache[config_id]["restarted"] += 1
        else:
            self._cache[config_id] = {
                "started": None,
                "restarted": 0,
                "accessed": 0,
                "inactive": 0
            }
            config = self.get_object("config", int(config_id))

            self.llm_pool.prepare_llm(config.config, config_id)
            self.llm_pool.start(config_id)
            self._cache[config_id]["started"] = dt.now()
        return config_id

    def unload_instance(self, config_id: Union[str, int]) -> Optional[str]:
        """
        Method for unloading a configured language model instance.
        :param config_id: Config ID.
        :return: Config ID if process as successful.
        """
        config_id = str(config_id)
        if config_id in self._cache:
            if self.llm_pool.is_running(config_id):
                self.llm_pool.stop(config_id)
            return config_id
        else:
            return None

    def forward_generate(self, config_id: Union[str, int], prompt: str) -> Optional[str]:
        """
        Method for forwarding a generate request to an instance.
        :param config_id: Config ID.
        :param prompt: Prompt.
        :return: Config ID.
        """
        config_id = str(config_id)
        self.load_instance(config_id)
        return self.llm_pool.generate(config_id, prompt)

    """
    Knowledgebase handling methods
    """

    def embed_via_instance(self, config_id: Union[str, int], documents: List[str]) -> List[Any]:
        """
        Wrapper method for embedding via instance.
        :param config_id: LLM config ID.
        :param documents: List of documents to embed.
        :return: List of embeddings.
        """
        embeddings = []
        for document in documents:
            embeddings.append(self.forward_generate(config_id, document))
        return embeddings

    def register_knowledgebase(self, kb_config_id: Union[str, int], embedding_config_id: Union[str, int]) -> str:
        """
        Method for registering knowledgebase.
        :param kb_config_id: Config ID for the knowledgebase.
        :param embedding_config_id: Config ID for the embedding model.
        :return: Config ID.
        """
        kb_config_id = str(kb_config_id)
        kb_config = self.get_object("config", int(kb_config_id))
        embedding_config_id = str(embedding_config_id)

        self.load_instance(embedding_config_id)

        handler = kb_config.pop("handler")
        handler_kwargs = {
            "peristant_directory": kb_config.pop("peristant_directory"),
            "metadata": kb_config.pop("metadata"),
            "base_embedding_function": None if embedding_config_id is None else lambda x: self.embed_via_instance(embedding_config_id, x),
            "implementation": kb_config.pop("implementation")
        }

        self.kbs[kb_config_id] = {"chromadb": ChromaKnowledgeBase}[handler](
            **handler_kwargs
        )
        return kb_config_id

    def create_default_knowledgebase(self, sub_path: str) -> int:
        """
        Method for creating default knowledgebase.
        :param sub_path: Sub path to locate knowledgebase under.
        :return: Knowledgebase config ID.
        """
        kb_id = self.post_object("knowledgebase",
                                 persistant_directory=os.path.join(self.knowledgebase_directory, sub_path))
        self.register_knowledgebase(kb_id)

    def delete_documents(self, kb_config_id: Union[str, int], document_ids: List[Any], collection: str = "base") -> None:
        """
        Method for deleting a document from the knowledgebase.
        :param kb_config_id: Config ID for the knowledgebase.
        :param document_ids: Document IDs.
        :param collection: Collection to remove document from.
        """
        for document_id in document_ids:
            self.kbs[str(kb_config_id)].delete_document(
                document_id, collection)

    def wipe_knowledgebase(self, kb_config_id: Union[str, int]) -> None:
        """
        Method for wiping a knowledgebase.
        :param kb_config_id: Config ID for the knowledgebase.
        """
        self.kbs[str(kb_config_id)].wipe_knowledgebase()

    def migrate_knowledgebase(self, source_config_id: Union[str, int], target_config_id: Union[str, int]) -> None:
        """
        Method for migrating knowledgebase.
        :param source_config_id: Config ID for the knowledgebase.
        :param target_config_id: Config ID for the knowledgebase.
        """
        pass

    def embed_documents(self, kb_config_id: Union[str, int], documents: List[str], metadatas: List[dict] = None, ids: List[str] = None, hashes: List[str] = None, collection: str = "base", compute_metadata: bool = False) -> None:
        """
        Method for embedding documents.
        :param kb_config_id: Config ID for the knowledgebase.
        :param documents: Documents to embed.
        :param metadatas: Metadata entries for documents.
            Defaults to None.
        :param ids: Custom IDs to add. 
            Defaults to the hash of the document contents.
        :param hashes: Content hashes.
            Defaults to None in which case hashes are computet.
        :param collection: Target collection.
            Defaults to "base".
        :param compute_metadata: Flag for declaring, whether to compute metadata.
            Defaults to False.
        """
        hashes = [hash_text_with_sha256(document.page_content)
                  for document in documents] if hashes is None else hashes
        for doc_index, hash in enumerate(hashes):
            if hash not in self.documents:
                path = os.path.join(self.document_directory, f"{hash}.bin")
                open(os.path.join(self.document_directory, f"{hash}.bin"), "wb").write(
                    documents[doc_index].encode("utf-8"))
                self.documents[hash] = {
                } if metadatas is None else metadatas[doc_index]
                self.documents[hash]["controller_library_path"] = path

        if metadatas is None:
            metadatas = [self.documents[hash] for hash in hashes]

        self.kbs[str(kb_config_id)].embed_documents(
            collection=collection, documents=documents, metadatas=metadatas, ids=hashes if ids is None else ids)

    """
    Custom methods
    """

    def forward_document_qa(self, llm_id: Union[int, str], kb_id: Union[int, str], query: str, include_sources: bool = True) -> dict:
        """
        Method for posting query.
        :param llm_id: LLM ID.
        :param kb_id: Knowledgebase ID.
        :param query: Query.
        :param include_sources: Flag declaring, whether to include sources.
        :return: Response.
        """
        docs = self.kbs[kb_id].get_retriever(
        ).get_relevant_documents(query=query)

        document_list = "'''" + "\n\n '''".join(
            [doc.page_content for doc in docs]) + "'''"
        generation_prompt = f"Answer the question '''{query}''' with the following information: \n\n {document_list}"

        response = self.forward_generate(llm_id, generation_prompt)

        return response, [doc.metadata for doc in docs] if include_sources else []
