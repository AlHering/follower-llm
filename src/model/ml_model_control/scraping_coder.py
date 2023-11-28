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
                 model_kwargs: dict = {},
                 tokenizer_path: str = None,
                 tokenizer_kwargs: dict = {},
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
        if backend == "ctransformers":
            from ctransformers import AutoModelForCausalLM as CAutoModelForCausalLM

            self.model = CAutoModelForCausalLM.from_pretrained(
                model_path_or_repo_id=model_path, model_file=model_file, **model_kwargs)
        elif backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path, **tokenizer_kwargs) if tokenizer_path is not None else None
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path, **model_kwargs)
        elif backend == "llamacpp":
            from llama_cpp import Llama

            self.model = Llama(model_path=os.path.join(
                model_path, model_file), **model_kwargs)
        elif backend == "autogptq":
            from transformers import AutoTokenizer
            from auto_gptq import AutoGPTQForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, **tokenizer_kwargs) if tokenizer_path is not None else None
            self.model = AutoGPTQForCausalLM.from_quantized(
                model_path, **model_kwargs)
        elif backend == "exllamav2":
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
        elif backend == "langchain_llamacpp":
            from langchain.llms import LlamaCpp

            self.model = LlamaCpp(model_path, **model_kwargs)

    def generate(self,
                 prompt: str,
                 history: List[Tuple[str, str]] = None,
                 generation_kwargs: dict = {}) -> str:
        """
        Method for generating a response to a given prompt and conversation history.
        :param prompt: Prompt.
        :param history: List of tuples of role ("system", "user", "assistant") and message.
        :param generation_kwargs: Generation kwargs as dictionary.
        """
        if history is None:
            history = [("system", self.system_prompt)]
        history.append(("user", prompt))
        full_prompt = "\n".join(
            f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n"
        if self.backend == "ctransformers" or self.backend == "langchain_llamacpp":
            answer = self.model(full_prompt, **generation_kwargs)
        elif self.backend == "transformers" or self.backend == "autogptq":
            input_tokens = self.tokenizer(
                full_prompt, return_tensors="pt").to(self.model.device)
            output_tokens = self.model.generate(
                **input_tokens, **generation_kwargs)[0]
            answer = self.tokenizer.decode(
                output_tokens, skip_special_tokens=True)
        elif self.backend == "llamacpp":
            answer = self.model(full_prompt, **generation_kwargs)
        elif self.backend == "exllamav2":
            answer = self.model.generate_simple(
                full_prompt, **generation_kwargs)
        return answer
