# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import glob
import torch
from typing import List, Tuple


class ScrapingCoder(object):
    """
    Class, representing Scraping Coders which utilize language models to support programming scraping infrastructure.
    """

    def __init__(self,
                 model_path: str,
                 backend: str,
                 model_file: str = None,
                 model_kwargs: dict = None,
                 tokenizer_path: str = None,
                 tokenizer_kwargs: dict = None,
                 default_system_prompt: str = "You are a friendly and helpful assistant answering questions based on the context provided.") -> None:
        """
        Initiation method.
        :param model_path: Path to model files.
        :param backend: Backend for model loading.
        :param model_file: Model file to load.
        :param model_kwargs: Model loading kwargs as dictionary.
        :param tokenizer_path: Tokenizer path.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
        :param default_system_prompt: Default system prompt.
        """
        self.backend = backend
        self.system_prompt = default_system_prompt
        self.tokenizer = None
        self.model = None

        initiation_kwargs = {
            "model_file": model_file, "model_kwargs": model_kwargs, "tokenizer_path": tokenizer_path, "tokenizer_kwargs": tokenizer_kwargs
        }
        {
            "ctransformers": self._initiate_ctransformers,
            "transformers": self._initiate_transformers,
            "llamacpp": self._initiate_llamacpp,
            "autogptq": self._initiate_autogptq,
            "exllamav2": self._initiate_exllamav2,
            "langchain_llamacpp": self._initiate_langchain_llamacpp
        }[backend](
            model_path=model_path,
            **{k: v for k, v in initiation_kwargs.items() if v is not None}
        )

    """
    Initiation methods
    """

    def _initiate_ctransformers(self,
                                model_path: str,
                                model_file: str = None,
                                model_kwargs: dict = {},
                                tokenizer_path: str = None,
                                tokenizer_kwargs: dict = {}) -> None:
        """
        Method for initiating ctransformers based tokenizer and model.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
        :param model_kwargs: Model loading kwargs as dictionary.
        :param tokenizer_path: Tokenizer path.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
        """
        from ctransformers import AutoModelForCausalLM as CAutoModelForCausalLM, AutoTokenizer as CAutoTokenizer
        self.model = CAutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=model_path, model_file=model_file, **model_kwargs)
        if tokenizer_path is not None:
            if tokenizer_path == model_path:
                self.tokenizer = CAutoTokenizer.from_pretrained(
                    self.model, **tokenizer_kwargs)
            else:
                self.tokenizer = CAutoTokenizer.from_pretrained(
                    tokenizer_path, **tokenizer_kwargs)

    def _initiate_transformers(self,
                               model_path: str,
                               model_file: str = None,
                               model_kwargs: dict = {},
                               tokenizer_path: str = None,
                               tokenizer_kwargs: dict = {}) -> None:
        """
        Method for initiating transformers based tokenizer and model.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
        :param model_kwargs: Model loading kwargs as dictionary.
        :param tokenizer_path: Tokenizer path.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path, **tokenizer_kwargs) if tokenizer_path is not None else None
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path, **model_kwargs)

    def _initiate_llamacpp(self,
                           model_path: str,
                           model_file: str = None,
                           model_kwargs: dict = {},
                           tokenizer_path: str = None,
                           tokenizer_kwargs: dict = {}) -> None:
        """
        Method for initiating llamacpp based tokenizer and model.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
        :param model_kwargs: Model loading kwargs as dictionary.
        :param tokenizer_path: Tokenizer path.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
        """
        try:
            from llama_cpp_cuda import Llama
        except ImportError:
            from llama_cpp import Llama

        self.model = Llama(model_path=os.path.join(
            model_path, model_file), **model_kwargs)
        self.model.__del__ = lambda _: None

    def _initiate_autogptq(self,
                           model_path: str,
                           model_file: str = None,
                           model_kwargs: dict = {},
                           tokenizer_path: str = None,
                           tokenizer_kwargs: dict = {}) -> None:
        """
        Method for initiating autogptq based tokenizer and model.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
        :param model_kwargs: Model loading kwargs as dictionary.
        :param tokenizer_path: Tokenizer path.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
        """
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, **tokenizer_kwargs) if tokenizer_path is not None else None
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_path, **model_kwargs)

    def _initiate_exllamav2(self,
                            model_path: str,
                            model_file: str = None,
                            model_kwargs: dict = {},
                            tokenizer_path: str = None,
                            tokenizer_kwargs: dict = {}) -> None:
        """
        Method for initiating exllamav2 based tokenizer and model.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
        :param model_kwargs: Model loading kwargs as dictionary.
        :param tokenizer_path: Tokenizer path.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
        """
        from exllamav2.model import ExLlamaV2, ExLlamaV2Cache, ExLlamaConfig, ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2BaseGenerator

        config = ExLlamaConfig(os.path.join(model_path, "config.json"))
        config.model_path = glob.glob(
            os.path.join(model_path, "*.safetensors"))
        model = ExLlamaV2(config)
        tokenizer = ExLlamaV2Tokenizer(
            os.path.join(model_path, "tokenizer.model"))
        cache = ExLlamaV2Cache(model)
        self.model = ExLlamaV2BaseGenerator(model, tokenizer, cache)

    def _initiate_langchain_llamacpp(self,
                                     model_path: str,
                                     model_file: str = None,
                                     model_kwargs: dict = {},
                                     tokenizer_path: str = None,
                                     tokenizer_kwargs: dict = {}) -> None:
        """
        Method for initiating langchain-llamacpp based tokenizer and model.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
        :param model_kwargs: Model loading kwargs as dictionary.
        :param tokenizer_path: Tokenizer path.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
        """
        from langchain.llms import LlamaCpp

        self.model = LlamaCpp(model_path, **model_kwargs)

    """
    Generation methods
    """

    def generate(self,
                 prompt: str,
                 history: List[Tuple[str, str]] = None,
                 generation_kwargs: dict = {}) -> Tuple[str, dict]:
        """
        Method for generating a response to a given prompt and conversation history.
        :param prompt: Prompt.
        :param history: List of tuples of role ("system", "user", "assistant") and message.
        :param generation_kwargs: Generation kwargs as dictionary.
        :return: Tuple of textual answer and metadata.
        """
        if history is None:
            history = [("system", self.system_prompt)]
        history.append(("user", prompt))
        full_prompt = "\n".join(
            f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n"

        metadata = None
        answer = None

        if self.backend == "ctransformers" or self.backend == "langchain_llamacpp":
            metadata = self.model(full_prompt, **generation_kwargs)
        elif self.backend == "transformers" or self.backend == "autogptq":
            input_tokens = self.tokenizer(
                full_prompt, return_tensors="pt").to(self.model.device)
            output_tokens = self.model.generate(
                **input_tokens, **generation_kwargs)[0]
            metadata = self.tokenizer.decode(
                output_tokens, skip_special_tokens=True)
        elif self.backend == "llamacpp":
            metadata = self.model(full_prompt, **generation_kwargs)
            answer = metadata["choices"][0]["text"]
        elif self.backend == "exllamav2":
            metadata = self.model.generate_simple(
                full_prompt, **generation_kwargs)

        return answer, metadata
