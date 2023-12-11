# -*- coding: utf-8 -*-
"""
****************************************************
*                   Follower LLM                   *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import Any, Tuple
from abc import ABC, abstractmethod


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
    def check_connection(self, source_metadata: dict = None) -> bool:
        """
        Method for checking connection.
        :param source_metadata: Scraping metadata for source.
        :return: True, if connection could be established, else False.
        """
        pass

    @abstractmethod
    def get_channel_info(self, channel_url: str, channel_metadata: dict = None) -> dict:
        """
        Method for acquiring channel info.
        :param channel_url: Channel URL.
        :param channel_metadata: Scraping metadata for channel.
        :return: Channel info.
        """
        pass

    @abstractmethod
    def get_asset_info(self, asset_url: str, asset_metadata: dict = None) -> dict:
        """
        Method for acquiring asset info.
        :param asset_url: Asset URL.
        :param asset_metadata: Scraping metadata for asset.
        :return: Asset info.
        """
        pass

    @abstractmethod
    def download_asset(self, asset_url: str, asset_metadata: dict = None) -> Tuple[dict, bytes]:
        """
        Method for downloading asset.
        :param asset_url: Asset URL.
        :param asset_metadata: Scraping metadata for asset.
        :return: Asset header and content.
        """
        pass
