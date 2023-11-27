# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import torch
from typing import List, Tuple
from transformers import LlamaForCausalLM, LlamaTokenizer
from src.configuration import configuration as cfg


class ScrapingCoder(object):
    """
    Class, representing Scraping Coders which utilize language models to support programming scraping infrastructure.
    """

    def __init__(self,
                 model_path: str,
                 model_file: str,
                 model_kwargs: dict = {
                     "torch_dtype": torch.float16, "device_map": "auto"},
                 tokenizer_kwargs: dict = {},
                 default_system_prompt: str = "") -> None:
        """
        Initiation method.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
        :param model_kwargs: Model loading kwargs as dictionary.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
        :param default_system_prompt: Default system prompt.
        """
        self.tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path, **tokenizer_kwargs)
        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path, model_file=model_file, **model_kwargs)
        self.system_prompt = default_system_prompt

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
        input_tokens = self.tokenizer(
            full_prompt, return_tensors="pt").to(self.model.device)
        output_tokens = self.model.generate(
            **input_tokens, **generation_kwargs)[0]
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    # https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF
    coder = ScrapingCoder(
        model_path=os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
        model_file="openhermes-2.5-mistral-7b-16k.Q6_K.gguf",
        default_system_prompt="You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
    )

    coder.generate(
        "Create a Python script for scraping the first 10 google hits for a search query.")
