# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from time import sleep
import traceback
import copy
from datetime import datetime as dt
from typing import Optional, Any, List, Dict, Union, Tuple, Callable
from src.configuration import configuration as cfg
from src.utility.gold.basic_sqlalchemy_interface import BasicSQLAlchemyInterface, FilterMask as FilterMask
from src.utility.bronze import sqlalchemy_utility
from src.model.scraping.data_model import populate_data_instrastructure
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utility.silver.file_system_utility import safely_create_path
from src.model.scraping.connector import Connector


class ScrapingController(BasicSQLAlchemyInterface):
    """
    Controller class for handling scraping operations.
    """

    def __init__(self, working_directory: str = None, database_uri: str = None, scraping_connectors: List[Connector] = None) -> None:
        """
        Initiation method.
        :param working_directory: Working directory.
            Defaults to folder 'processes' folder under standard backend data path.
        :param database_uri: Database URI.
            Defaults to 'backend.db' file under default data path.
        :param scraping_connectors: Connectors for scraping operation.
            Defaults to None in which case the connector list will be empty at start.
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

        # Scraping infrastructure
        self.document_directory = os.path.join(
            self.working_directory, "library")
        safely_create_path(self.document_directory)

        # Cache
        self._cache = {
            "cns": {}
        }
        for connector in scraping_connectors:
            self.register_connector(connector)

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

    def start_up(self) -> None:
        """
        Method for running startup process.
        """
        pass

    """
    Exit and shutdown methods
    """

    def shutdown(self) -> None:
        """
        Method for running shutdown process.
        """
        pass

    """
    Scraping connector handling methods
    """

    def register_connector(self, connector: Connector) -> bool:
        """
        Method for registering an connector.
        :param connector: Connector to register.
        :return: True, if process was successful else False.
        """
        try:
            source = connector.get_source_name()
            if source not in self._cache["cns"]:
                self._logger.info(
                    f"Registered new connector for '{source}'")
            else:
                self._logger.warn(
                    f"Connector for '{source}' was overwritten")
            self._cache["cns"][source] = connector
            return True
        except Exception as ex:
            self._logger.warn(
                f"Exception appeared while trying to register an connector for '{source}': {ex}\nTrace: {traceback.format_exc()}")
            return False

    def registration_gateway(self, object_type: str, object_attributes: dict, source_id: int = None, parent_id_attribute: str = None, parent_id: int = None) -> None:
        """
        Method for cleaning object attributes for a given object type and registering the object.
        :param object_type: Object type.
        :param object_attributes: Object attributes.
        :param source_id: Source ID.
            Defaults to None.
        :param parent_id_attribute: Parent ID attribute.
            Defaults to None.
        :param parent_id: Parent object ID.
            Defaults to None.
        """
        if source_id is not None:
            object_attributes["source_id"] = source_id
        if parent_id_attribute is not None and parent_id is not None:
            object_attributes[parent_id_attribute] = parent_id
        self.put_object(object_type, reference_attributes=[
                        "url"], **{key: object_attributes[key] for key in object_attributes if hasattr(self.model[object_type], key)})

    def _start_scraping_threads(self, connector: Connector, scraping_batch: List[Tuple[str, Any, dict, List[Callable]]]) -> List[dict]:
        """
        Method for starting a scraping thread for a given scraping batch.
        :param connector: Connector.
        :param scraping_batch: List of tuples of target type, the scraping target object,
            an scraping metadata and a list of callback functions.
        :return: Scraping report.
        """

        connector_methods = {
            "feed": connector.scrape_feed,
            "channel": connector.scrape_channel,
            "asset": connector.scrape_asset
        }
        entries = {
            scraping_entry[1].url: scraping_entry for scraping_entry in scraping_batch
        }
        jobs = {}
        reports = [{"status": "failed",
                    "url": scraping_entry[1].url,
                    "reason": f"Error in scraping process."} for scraping_entry in scraping_batch]

        with ThreadPoolExecutor(max_workers=20) as thread_executor:
            for entry_url in enumerate(entries):
                jobs[entry_url] = thread_executor.submit(
                    connector_methods[entries[entry_url][0]], entry_url, entries[entry_url][2], *
                    entries[entry_url][3]
                )

            for future in as_completed(jobs):
                result = jobs[future].result()
                reports[scraping_batch.index(entries[future])] = {
                    "status": "successful",
                    "url": future,
                    "result": result
                }
                self.put_object(entries[future][0], [
                                "url"], url=future, info=result)
        return reports

    """ 
    Interaction methods
    """

    def scrape_by_anchor(self, source_id: int, anchor: str = "feed", anchor_ids: List[int] = None, start_time: dt = None, end_time: dt = None) -> List[dict]:
        """
        Method for scraping by anchor.
        :param source_id: Source ID.
        :param anchor: Scraping anchor as string. Should be one of "feed", "channel", "asset".
            Defaults to "feed".
        :param anchor_ids: IDs of the anchor objects to scrape.
            Defaults to None in which case all available anchors are scraped.
        :param start_time: Timestamp for declaring a datetime as lower bound for scraping.
            Defaults to None in which case the all available entries are scraped.
        :param end_time: Timestamp for declaring a datetime as upper bound for scraping.
            Defaults to None in which case the current datetime is choosen as upper bound.
        :return: Scraping reports.
        """
        self._logger.info(
            f"Preparing scraping process for '{anchor}' with IDs '{anchor_ids}'")
        source = self.get_object_by_id("source", source_id)
        if source is None:
            return [{"status": "failed",
                     "source_url": source_id,
                     "anchor": anchor,
                     "reason": f"Could not find database entry for source ID '{source_id}'"}]

        if anchor_ids is not None:
            targets = [self.get_object_by_id(
                anchor, anchor_id) for anchor_id in anchor_ids]
            if not targets:
                return [{"status": "failed", "reason": f"No targets found for source '{source}' and ID '{anchor_id}'"} for anchor_id in anchor_ids]
        else:
            targets = self.get_objects_by_filtermasks(
                anchor, [FilterMask(["source_id", "==", source_id])])
            if not targets:
                return [{"status": "failed", "reason": f"No targets found for source '{source}'"}]

        connector: Connector = self._cache["cns"].get(source.name)
        if connector is None:
            return [{"status": "failed",
                    "source_name": source.name,
                     "source_url": source.url,
                     "target_type": anchor,
                     "reason": f"Could not find connector for target source"}]

        scraping_metadata_update = {}
        if start_time is not None:
            scraping_metadata_update["start_time"] = start_time
        if end_time is not None:
            scraping_metadata_update["end_time"] = end_time

        if anchor == "feed":
            callbacks = [lambda _, object_attributes: self.registration_gateway("channel", object_attributes, source_id),
                         lambda _, object_attributes: self.registration_gateway("asset", object_attributes, source_id)]
        elif anchor == "channel":
            callbacks = [lambda parent_id, object_attributes: self.registration_gateway(
                "asset", object_attributes, source_id, "channel_id", parent_id)]
        elif anchor == "asset":
            callbacks = [lambda parent_id, object_attributes: self.registration_gateway(
                "file", object_attributes, None, "asset_id", parent_id)]

        scraping_batches = []
        for target in targets:
            scraping_metadata = target.scraping_metadata
            scraping_metadata.update(scraping_metadata_update)

            scraping_batches.append(
                (anchor, target, copy.deepcopy(scraping_metadata), [lambda object_attributes: callback(target.id, object_attributes) for callback in callbacks]))

        return self._start_scraping_threads(connector=connector,
                                            scraping_batches=scraping_batches)
