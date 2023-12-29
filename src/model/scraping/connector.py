# -*- coding: utf-8 -*-
"""
****************************************************
*                   Follower LLM                   *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from typing import Any, Tuple, Union
from abc import ABC, abstractmethod
import traceback
from src.configuration import configuration as cfg
from src.utility.silver.environment_utility import get_lambda_function_from_string
from src.utility.bronze.requests_utility import download_web_asset
from urllib.parse import urlparse
from lxml import html
import json
import requests


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

    def validate_url_responsiblity(self, url: str) -> bool:
        """
        Method for validating the responsiblity for a URL.
        :param url: Target URL.
        :return: True, if connector is responsible for URL else False.
        """
        return urlparse(url).netloc in self.base_url

    @abstractmethod
    def scrape_feed(self, feed_url: str, feed_metadata: dict = None, channel_registration_callback: Any = None, asset_registration_callback: Any = None) -> dict:
        """
        Method for acquiring feed info and register contained channels or assets if a callback is given.
        :param feed_url: Feed URL.
        :param feed_metadata: Scraping metadata for feed.
        :param channel_registration_callback: Callback function for registering channels.
        :param asset_registration_callback: Callback function for registering assets.
        :return: Feed info.
        """
        pass

    @abstractmethod
    def scrape_channel(self, channel_url: str, channel_metadata: dict = None, registration_callback: Any = None) -> dict:
        """
        Method for acquiring channel info and register contained assets if a callback is given.
        :param channel_url: Channel URL.
        :param channel_metadata: Scraping metadata for channel.
        :param registration_callback: Callback function for registering assets.
        :return: Channel info.
        """
        pass

    @abstractmethod
    def scrape_asset(self, asset_url: str, asset_metadata: dict = None, registration_callback: Any = None) -> dict:
        """
        Method for acquiring asset info and register contained files if a callback is given.
        :param asset_url: Asset URL.
        :param asset_metadata: Scraping metadata for asset.
        :param registration_callback: Callback function for registering files.
        :return: Asset info.
        """
        pass

    @abstractmethod
    def download_asset(self, output_path: str, asset_url: str, asset_metadata: dict = None) -> bool:
        """
        Method for downloading asset files.
        :param output_path: Output path.
        :param asset_url: Asset URL.
        :param asset_metadata: Scraping metadata for asset.
        :return: True, if process was successfull else False.
        """
        pass


class ProfileConnector(Connector):
    """
    Class, representing a profiled scraping connector.
    """

    def __init__(self, profile: dict) -> None:
        """
        Initiation method.
        :param profile: Profile dictionary, containing
            'source': Source name.
            'base_url': Base URL.
            'authorization': Authorization key field in environment or authorization key.

            'feed_info_parser': Lambda expression for extracting feed info from response content.
                Can be given as string.
            'channel_info_parser': Lambda expression for extracting channel info from response content.
                Can be given as string.
            'asset_info_parser': Lambda expression for extracting asset info from response content.
                Can be given as string.

            'channel_extractor': Lambda expression for extracting asset channel registration entries from response content.
                Can be given as string.
            'asset_extractor': Lambda expression for extracting asset asset registration entries from response content.
                Can be given as string.
            'file_extractor': Lambda expression for extracting asset file registration entries from response content.
                Can be given as string.
        """
        self._logger = cfg.LOGGER
        self.profile = profile
        self.source = self.profile["source"]
        self.base_url = self.profile["base_url"]
        self.authorization = cfg.ENV.get(
            self.profile["authorization"], self.profile["authorization"])

        # Info parsers
        for parser in ["feed_info_parser", "channel_info_parser", "asset_info_parser"]:
            setattr(self, parser, get_lambda_function_from_string(self.profile[parser]) if isinstance(
                self.profile[parser], str) else self.profile[parser])

        # Registration extractors
        for extractor in ["channel_extractor", "asset_extractor", "file_extractor"]:
            setattr(self, extractor, get_lambda_function_from_string(self.profile[extractor]) if isinstance(
                self.profile[extractor], str) else self.profile[extractor])

    def get_source_name(self) -> str:
        """
        Method for acquiring the source name.
        :return: Source name.
        """
        return self.source

    def check_connection(self, source_metadata: dict = None) -> bool:
        """
        Method for checking connection.
        :param source_metadata: Scraping metadata for source.
        :return: True, if connection could be established, else False.
        """
        result = requests.get(self.base_url).status_code == 200
        self._logger.info("Connection was successfuly established.") if result else self._logger.warn(
            "Connection could not be established.")
        return result

    def validate_url_responsiblity(self, url: str) -> bool:
        """
        Method for validating the responsiblity for a URL.
        :param url: Target URL.
        :param kwargs: Arbitrary keyword arguments.
        :return: True, if wrapper is responsible for URL else False.
        """
        return urlparse(url).netloc in self.base_url

    def _build_request_from_metadata(self, url: str, metadata: dict) -> Union[html.HtmlElement, dict]:
        """
        Method for building request from metadata.
        :param url: Target URL.
        :param metadata: Scraping metadata.
        """
        response = requests.get(
            url=url,
            headers=metadata.get("headers"),
            proxies=metadata.get("proxies"),
            cookies=metadata.get("cookies"),
            stream=metadata.get("stream"),
            verify=metadata.get("verify"),
            cert=metadata.get("cert")
        )
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return html.fromstring(response.content)

    # Override
    def scrape_feed(self, feed_url: str, feed_metadata: dict = None, channel_registration_callback: Any = None, asset_registration_callback: Any = None) -> dict:
        """
        Method for acquiring feed info and register contained channels or assets if a callback is given.
        :param feed_url: Feed URL.
        :param feed_metadata: Scraping metadata for feed.
        :param channel_registration_callback: Callback function for registering channels.
        :param asset_registration_callback: Callback function for registering assets.
        :return: Feed info.
        """
        html = self._build_request_from_metadata(feed_url, feed_metadata)
        if channel_registration_callback is not None:
            for entry in self.channel_extractor(html):
                channel_registration_callback(entry)
        if asset_registration_callback is not None:
            for entry in self.asset_extractor(html):
                asset_registration_callback(entry)
        return self.feed_info_parser(html)

    # Override

    def scrape_channel(self, channel_url: str, channel_metadata: dict = None, registration_callback: Any = None) -> dict:
        """
        Method for acquiring channel info and register contained assets if a callback is given.
        :param channel_url: Channel URL.
        :param channel_metadata: Scraping metadata for channel.
        :param registration_callback: Callback function for registering assets.
        :return: Channel info.
        """
        html = self._build_request_from_metadata(channel_url, channel_metadata)
        if registration_callback is not None:
            for entry in self.asset_extractor(html):
                registration_callback(entry)
        return self.channel_info_parser(html)

    # Override
    def scrape_asset(self, asset_url: str, asset_metadata: dict = None, registration_callback: Any = None) -> dict:
        """
        Method for acquiring asset info and register contained files if a callback is given.
        :param asset_url: Asset URL.
        :param asset_metadata: Scraping metadata for asset.
        :param registration_callback: Callback function for registering files.
        :return: Asset info.
        """
        html = self._build_request_from_metadata(asset_url, asset_metadata)
        if registration_callback is not None:
            for entry in self.file_extractor(html):
                registration_callback(entry)
        return self.asset_info_parser(html)

    # Override
    def download_asset(self, output_path: str, asset_url: str, asset_metadata: dict = None) -> bool:
        """
        Method for downloading asset.
        :param output_path: Output path.
        :param asset_url: Asset URL.
        :param asset_metadata: Scraping metadata for asset.
        :return: True, if process was successfull else False.
        """
        try:
            download_web_asset(asset_url=asset_url,
                               output_path=output_path,
                               add_extension=asset_metadata.get(
                                   "add_extension", False),
                               headers=asset_metadata.get("headers"))
            return True
        except Exception as ex:
            self._logger.warn(f"Exception appeared: {ex}")
            self._logger.warn(f"Trace: {traceback.format_exc()}")
            return False
