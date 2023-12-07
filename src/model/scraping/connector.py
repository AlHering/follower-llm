# -*- coding: utf-8 -*-
"""
****************************************************
*                   Follower LLM                   *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import Any, Tuple
from abc import ABC, abstractmethod
from src.utility.silver import environment_utility
from src.model.plugin_model.plugins import GenericPlugin, PluginImportException, PluginRuntimeException


class Connector(ABC):
    """
    Class, representing a scraping connector.
    """

    @abstractmethod
    def get_source_name(self) -> str:
        """
        Method for acquiring the source name.
        :return: Source name.
        """
        pass

    @abstractmethod
    def check_connection(self) -> bool:
        """
        Method for checking connection.
        :return: True, if connection could be established, else False.
        """
        pass

    @abstractmethod
    def get_channel_info(self, channel_id: Any) -> dict:
        """
        Method for acquiring channel info.
        :param channel_id: Channel ID.
        :return: Channel info.
        """
        pass

    @abstractmethod
    def get_asset_info(self, channel_id: Any, asset_id: Any) -> dict:
        """
        Method for acquiring asset info.
        :param channel_id: Channel ID.
        :param asset_id: Asset ID.
        :return: Asset info.
        """
        pass

    @abstractmethod
    def download_asset(self, channel_id: Any, asset_id: Any) -> Tuple[dict, bytes]:
        """
        Method for downloading asset.
        :param channel_id: Channel ID.
        :param asset_id: Asset ID.
        :return: Asset header and content.
        """
        pass
