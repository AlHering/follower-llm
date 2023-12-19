# -*- coding: utf-8 -*-
"""
****************************************************
*                   Follower LLM                   *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from src.model.scraping.connector import Connector
from src.utility.silver import environment_utility
from src.model.plugin_model.plugins import GenericPlugin, PluginImportException, PluginRuntimeException


class ScrapingPlugin(GenericPlugin):
    """
    Class, representing a scraping plugin.
    """

    def __init__(self, info: dict, path: str, security_hash: str = None) -> None:
        """
        Representation of a scraping plugin for adding scraping connectors.
        :param info: Plugin info.
        :param path: Path to plugin.
        :param security_hash: Hash that can be used for authentication purposes.
            Defaults to None.
        """
        super().__init__(info, path, security_hash)
        if "./" in self.info["connector"]:
            self.info["connector"] = os.path.normpath(
                os.path.join(path, self.info["blueprints"]))
        self.connector: Connector = environment_utility.get_module(
            self.info["connector"], security_hash)
        for func in ["get_source_name", "check_connection", "get_channel_info", "get_asset_info", "download_asset"]:
            if not hasattr(self.connector, func):
                raise PluginImportException(
                    self.name, self.type, f"Plugin does not implement '{func}'-function correctly")
            setattr(self, func, lambda *args, **
                    kwargs: getattr(self.connector, func)(*args, **kwargs))

    def validate(self) -> None:
        """
        Method for validating plugin import.
        :param args: Arbitrary arguments.
        :param kwargs: Arbitrary keyword arguments.
        """
        if "connector" not in self.info:
            raise PluginImportException(
                "Plugin info does not contain a 'connector' value")
