# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from datetime import datetime as dt
from src.utility.gold.language_model_utility import Agent, LanguageModelInstance
from typing import List, Tuple, Any, Callable, Optional, Type


class LangchainScrapingCoder(Agent):
    """
    Class, representing Scraping Coders which utilize language models to support programming scraping infrastructure.
    """

    def __init__(self,
                 general_llm: LanguageModelInstance) -> None:
        """
        Initiation method.
        :param general_llm: LanguageModelInstance for general tasks.
        """
