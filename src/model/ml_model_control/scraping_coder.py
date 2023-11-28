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
from typing import List, Tuple, Any

"""
Model backend overview
------------------------------------------
llama-cpp-python - GGML/GGUF run on CPU, offload layers to GPU, CUBLAS support (CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python)
- CPU: llama-cpp-python==0.2.18
- GPU: https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.18+cu117-cp310-cp310-manylinux_2_31_x86_64.whl ; platform_system == "Linux" and platform_machine == "x86_64"

exllama - 4-bit GPTQ weights, GPU inference (tested on newer GPUs > Pascal)
- CPU: exllamav2==0.0.5
- GPU: https://github.com/jllllll/exllamav2/releases/download/v0.0.5/exllamav2-0.0.5+cu117-cp310-cp310-linux_x86_64.whl; platform_system == "Linux" and platform_machine == "x86_64"

auto-gptq - 4-bit GPTQ weights, GPU inference, can be used with Triton (auto-gptq[triton])
- CPU: auto-gptq==0.5.1
- GPU: https://github.com/jllllll/AutoGPTQ/releases/download/v0.5.1/auto_gptq-0.5.1+cu117-cp310-cp310-linux_x86_64.whl; platform_system == "Linux" and platform_machine == "x86_64"

gptq-for-llama - 4-bit GPTQ weights, GPU inference -> practically replaced by auto-gptq !
- CPU: gptq-for-llama==0.1.0
- GPU: https://github.com/jllllll/GPTQ-for-LLaMa-CUDA/releases/download/0.1.0/gptq_for_llama-0.1.0+cu117-cp310-cp310-linux_x86_64.whl; platform_system == "Linux" and platform_machine == "x86_64"

transformers - support common model architectures, CUDA support (e.g. via PyTorch)
- CPU: transformers==4.35.2
- GPU: use PyTorch e.g.:
    --extra-index-url https://download.pytorch.org/whl/cu117
    torch==2.0.1+cu117
    torchaudio==2.0.2+cu117
    torchvision==0.15.2+cu117

ctransformers - transformers C bindings, Cuda support (ctransformers[cuda])
- CPU: ctransformers==0.2.27
- GPU: ctransformers[cuda]==0.2.27 or https://github.com/jllllll/ctransformers-cuBLAS-wheels/releases/download/AVX2/ctransformers-0.2.27+cu117-py3-none-any.whl
"""


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
        self.config = None
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
        from ctransformers import AutoConfig as CAutoConfig, AutoModelForCausalLM as CAutoModelForCausalLM, AutoTokenizer as CAutoTokenizer

        self._update_config(CAutoConfig.from_pretrained(
            model_path_or_repo_id=model_path), model_kwargs=model_kwargs, overwrite_kwargs=True)
        self.model = CAutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=model_path, model_file=model_file, **model_kwargs)
        # TODO: Currently ctransformers' tokenizer from model is not working.
        if False and tokenizer_path is not None:
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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self._update_config(AutoConfig.from_pretrained(
            model_path_or_repo_id=model_path), model_kwargs=model_kwargs, overwrite_kwargs=True)
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
        from exllamav2.model import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
        from exllamav2.tokenizer import ExLlamaV2Tokenizer
        from exllamav2.generator import ExLlamaV2BaseGenerator

        config = ExLlamaV2Config(os.path.join(model_path, "config.json"))
        if model_file is None:
            config.model_path = config.model_path = glob.glob(
                os.path.join(model_path, "*.safetensors"))
        else:
            config.model_path = os.path.join(model_path, model_file)
        model = ExLlamaV2(config)
        tokenizer = ExLlamaV2Tokenizer(
            os.path.join(tokenizer_path, "tokenizer.model"))
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
        from langchain.llms.llamacpp import LlamaCpp

        if model_file is not None and not model_path.endswith(model_file):
            model_path = os.path.join(model_path, model_file)
        self.model = LlamaCpp(model_path=model_path, **model_kwargs)

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

    """
    Utility methods
    """

    def _update_config(self, config_obj: Any, model_kwargs, overwrite_kwargs: bool = True) -> None:
        """
        Method for updating the instance config from model kwargs.
        :param config_obj: Config object.
        :param model_kwargs: Model kwargs.
        :param overwrite_kwargs: Flag for declaring whether to replace model_kwargs["config"]
            with config object after update. Defaults to False.
        """
        if "config" in model_kwargs:
            self.config = config_obj
            for key in model_kwargs["config"]:
                setattr(self.config, key, model_kwargs["config"][key])
            if overwrite_kwargs:
                model_kwargs["config"] = self.config
