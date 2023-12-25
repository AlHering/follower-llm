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
from src.model.database.data_model import populate_data_instrastructure
from src.model.language_models.llm_pool import ThreadedLLMPool
from src.model.knowledgebase.knowledgebase_router import spawn_knowledgebase_instance
from src.utility.silver import embedding_utility
from src.utility.silver.file_system_utility import safely_create_path


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
        # TODO: Update default embeddings alongside new language model utility
        self.default_embedding_function = embedding_utility.LocalHuggingFaceEmbeddings(
            cfg.PATHS.INSTRUCT_XL_PATH
        )
        for kb in self.get_objects_by_type("kbinstance"):
            self.register_knowledgebase(backend=kb.backend,
                                        knowledgebase_path=kb.knowledgebase_path,
                                        knowledgebase_parameters=kb.knowledgebase_parameters,
                                        preprocessing_parameters=kb.preprocessing_parameters,
                                        embedding_parameters=kb.embedding_parameters,
                                        retrieval_parameters=kb.retrieval_parameters,
                                        embedding_model_instance_id=kb.embedding_model_instance_id,
                                        owner_id=kb.owner_id,
                                        create_database_entry=False)

        # LLM infrastructure
        self.llm_pool = ThreadedLLMPool()

        # Cache
        self._cache = {
            "lms": {},
            "kbs": {}
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
        while any(self.llm_pool.is_running(instance_id) for instance_id in self._cache["lms"]):
            sleep(2.0)

    """
    LLM handling methods
    """

    def load_instance(self, lm_instance_id: Union[str, int]) -> Optional[str]:
        """
        Method for loading a configured language model instance.
        :param lm_instance_id: Model instance ID.
        :return: Model instance ID if process as successful.
        """
        lm_instance_id = str(lm_instance_id)
        if lm_instance_id in self._cache["lms"]:
            if not self.llm_pool.is_running(lm_instance_id):
                self.llm_pool.start(lm_instance_id)
                self._cache["lms"][lm_instance_id]["restarted"] += 1
        else:
            self._cache["lms"][lm_instance_id] = {
                "started": None,
                "restarted": 0,
                "accessed": 0,
                "inactive": 0
            }
            obj = self.get_object("lminstance", int(lm_instance_id))

            self.llm_pool.prepare_llm({
                "backend": obj.backend,
                "model_path": obj.model_path,
                "model_file": obj.model_file,
                "model_parameters": obj.model_parameters,
                "tokenizer_path": obj.tokenizer_path,
                "tokenizer_parameters": obj.tokenizer_parameters,
                "config_path": obj.config_path,
                "config_parameters": obj.config_parameters,
                "default_system_prompt": obj.default_system_prompt,
                "use_history": obj.use_history,
                "encoding_parameters": obj.encoding_parameters,
                "generating_parameters": obj.generating_parameters,
                "decoding_parameters": obj.decoding_parameters
            }, lm_instance_id)
            self.llm_pool.start(lm_instance_id)
            self._cache["lms"][lm_instance_id]["started"] = dt.now()
        return lm_instance_id

    def unload_instance(self, lm_instance_id: Union[str, int]) -> Optional[str]:
        """
        Method for unloading a configured language model instance.
        :param lm_instance_id: Config ID.
        :return: Config ID if process as successful.
        """
        lm_instance_id = str(lm_instance_id)
        if lm_instance_id in self._cache["lms"]:
            if self.llm_pool.is_running(lm_instance_id):
                self.llm_pool.stop(lm_instance_id)
            return lm_instance_id
        else:
            return None

    def forward_generate(self, lm_instance_id: Union[str, int], prompt: str) -> Optional[str]:
        """
        Method for forwarding a generate request to an instance.
        :param lm_instance_id: Config ID.
        :param prompt: Prompt.
        :return: Config ID.
        """
        lm_instance_id = str(lm_instance_id)
        self.load_instance(lm_instance_id)
        return self.llm_pool.generate(lm_instance_id, prompt)

    """
    Knowledgebase handling methods
    """

    def register_knowledgebase(self,
                               backend: str,
                               knowledgebase_path: str,
                               knowledgebase_parameters: dict = None,
                               preprocessing_parameters: dict = None,
                               embedding_parameters: dict = None,
                               retrieval_parameters: dict = None,
                               embedding_model_instance_id: int = None,
                               owner_id: int = None,
                               kb_instance_id: int = None) -> str:
        """
        Method for registering knowledgebase.
        :param backend: Backend of the knowledgebase instance.
        :param knowledgebase_path: Path of the knowledgebase instance.
        :param knowledgebase_parameters: Parameters for the knowledgebase instantiation.
            Defaults to None.
        :param preprocessing_parameters: Parameters for document preprocessing.
            Defaults to None.
        :param embedding_parameters: Parameters for document embedding.
            Defaults to None.
        :param retrieval_parameters: Parameters for the document retrieval.
            Defaults to None.
        :param embedding_model_instance_id: ID of the model instance to use as default embedding model.
            Defaults to None.
        :param owner_id: ID of the creating user.
            Defaults to None.
        :param kb_instance_id: Knowledgebase instance ID.
            Defaults to None in which case a new database entry is created.
        """
        kb_instance_id = self.post_object("kbinstance",
                                          backend=backend,
                                          knowledgebase_path=knowledgebase_path,
                                          knowledgebase_parameters=knowledgebase_parameters,
                                          preprocessing_parameters=preprocessing_parameters,
                                          embedding_parameters=embedding_parameters,
                                          retrieval_parameters=retrieval_parameters,
                                          embedding_model_instance_id=embedding_model_instance_id,
                                          owner_id=owner_id) if kb_instance_id is None else kb_instance_id
        self._cache["kbs"][str(kb_instance_id)] = spawn_knowledgebase_instance(
            backend=backend,
            knowledgebase_path=knowledgebase_path,
            knowledgebase_parameters=knowledgebase_parameters,
            preprocessing_parameters=preprocessing_parameters,
            embedding_parameters=embedding_parameters,
            retrieval_parameters=retrieval_parameters,
            embedding_model_instance_id=embedding_model_instance_id
        )

    """
    Interaction methods
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
