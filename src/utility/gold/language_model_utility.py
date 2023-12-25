
# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import traceback
from pydantic import BaseModel
from typing import List, Tuple, Any, Callable, Optional, Type, Union
from uuid import uuid4
from datetime import datetime as dt
from src.configuration import configuration as cfg
from .filter_mask import FilterMask

# TODO: Plan out and implement common utility.
"""
Model backend overview
------------------------------------------
llama-cpp-python - GGML/GGUF run on CPU, offload layers to GPU, CUBLAS support (CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python)
- CPU: llama-cpp-python==0.2.18
- GPU: https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.18+cu117-cp310-cp310-manylinux_2_31_x86_64.whl ; platform_system == "Linux" and platform_machine == "x86_64"

exllamav2 - 4-bit GPTQ weights, GPU inference (tested on newer GPUs > Pascal)
- CPU: exllamav2==0.0.9
- GPU: https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.18+cu117-cp310-cp310-manylinux_2_31_x86_64.whl; platform_system == "Linux" and platform_machine == "x86_64"

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


"""
Model instantiation functions
"""


def load_ctransformers_model(model_path: str,
                             model_file: str = None,
                             model_parameters: dict = {},
                             tokenizer_path: str = None,
                             tokenizer_parameters: dict = {},
                             embeddings_path: str = None,
                             embeddings_parameters: dict = {},
                             config_path: str = None,
                             config_parameters: dict = {}) -> Tuple:
    """
    Function for loading ctransformers based model objects.
    :param model_path: Path to model files.
    :param model_file: Model file to load.
        Defaults to None.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param tokenizer_path: Tokenizer path.
        Defaults to None.
    :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param embeddings_path: Embeddings path.
        Defaults to None.
    :param embeddings_parameters: Embeddings loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param config_path: Config path.
        Defaults to None.
    :param config_parameters: Config loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Tuple of config, tokenizer, embeddings, model and generator object.
        Note, that all objects that do not belong to the backend will be None.
    """
    from ctransformers import AutoConfig as CAutoConfig, AutoModelForCausalLM as CAutoModelForCausalLM, AutoTokenizer as CAutoTokenizer

    config = None
    tokenizer = None
    embeddings = None
    model = None
    generator = None

    if config_path:
        config = CAutoConfig.from_pretrained(
            model_path_or_repo_id=config_path)
        if config_parameters is not None:
            for key in config_parameters:
                setattr(config, key, config_parameters[key])

    model = CAutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=model_path, model_file=model_file, config=config, **model_parameters)
    # TODO: Currently ctransformers' tokenizer from model is not working.
    if False and tokenizer_path is not None:
        if tokenizer_path == model_path:
            tokenizer = CAutoTokenizer.from_pretrained(
                model, **tokenizer_parameters)
        else:
            tokenizer = CAutoTokenizer.from_pretrained(
                tokenizer_path, **tokenizer_parameters)

    return (config, tokenizer, embeddings, model, generator)


def load_transformers_model(model_path: str,
                            model_file: str = None,
                            model_parameters: dict = {},
                            tokenizer_path: str = None,
                            tokenizer_parameters: dict = {},
                            embeddings_path: str = None,
                            embeddings_parameters: dict = {},
                            config_path: str = None,
                            config_parameters: dict = {}) -> Tuple:
    """
    Function for loading transformers based model objects.
    :param model_path: Path to model files.
    :param model_file: Model file to load.
        Defaults to None.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param tokenizer_path: Tokenizer path.
        Defaults to None.
    :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param embeddings_path: Embeddings path.
        Defaults to None.
    :param embeddings_parameters: Embeddings loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param config_path: Config path.
        Defaults to None.
    :param config_parameters: Config loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Tuple of config, tokenizer, embeddings, model and generator object.
        Note, that all objects that do not belong to the backend will be None.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = None
    tokenizer = None
    embeddings = None
    model = None
    generator = None

    if config_path:
        config = AutoConfig.from_pretrained(
            model_path_or_repo_id=config_path)
    if config_parameters is not None:
        for key in config_parameters:
            setattr(config, key, config_parameters[key])

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path, **tokenizer_parameters) if tokenizer_path is not None else None
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path, config=config, **model_parameters)

    return (config, tokenizer, embeddings, model, generator)


def load_llamacpp_model(model_path: str,
                        model_file: str = None,
                        model_parameters: dict = {},
                        tokenizer_path: str = None,
                        tokenizer_parameters: dict = {},
                        embeddings_path: str = None,
                        embeddings_parameters: dict = {},
                        config_path: str = None,
                        config_parameters: dict = {}) -> Tuple:
    """
    Function for loading llamacpp based model objects.
    :param model_path: Path to model files.
    :param model_file: Model file to load.
        Defaults to None.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param tokenizer_path: Tokenizer path.
        Defaults to None.
    :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param embeddings_path: Embeddings path.
        Defaults to None.
    :param embeddings_parameters: Embeddings loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param config_path: Config path.
        Defaults to None.
    :param config_parameters: Config loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Tuple of config, tokenizer, embeddings, model and generator object.
        Note, that all objects that do not belong to the backend will be None.
    """
    try:
        from llama_cpp_cuda import Llama
    except ImportError:
        from llama_cpp import Llama

    config = None
    tokenizer = None
    embeddings = None
    model = None
    generator = None

    model = Llama(model_path=os.path.join(
        model_path, model_file), **model_parameters)
    return (config, tokenizer, embeddings, model, generator)


def load_autogptq_model(model_path: str,
                        model_file: str = None,
                        model_parameters: dict = {},
                        tokenizer_path: str = None,
                        tokenizer_parameters: dict = {},
                        embeddings_path: str = None,
                        embeddings_parameters: dict = {},
                        config_path: str = None,
                        config_parameters: dict = {}) -> Tuple:
    """
    Function for loading autogptq based model objects.
    :param model_path: Path to model files.
    :param model_file: Model file to load.
        Defaults to None.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param tokenizer_path: Tokenizer path.
        Defaults to None.
    :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param embeddings_path: Embeddings path.
        Defaults to None.
    :param embeddings_parameters: Embeddings loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param config_path: Config path.
        Defaults to None.
    :param config_parameters: Config loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Tuple of config, tokenizer, embeddings, model and generator object.
        Note, that all objects that do not belong to the backend will be None.
    """
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM

    config = None
    tokenizer = None
    embeddings = None
    model = None
    generator = None

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, **tokenizer_parameters) if tokenizer_path is not None else None
    model = AutoGPTQForCausalLM.from_quantized(
        model_path, **model_parameters)
    return (config, tokenizer, embeddings, model, generator)


def load_exllamav2_model(model_path: str,
                         model_file: str = None,
                         model_parameters: dict = {},
                         tokenizer_path: str = None,
                         tokenizer_parameters: dict = {},
                         embeddings_path: str = None,
                         embeddings_parameters: dict = {},
                         config_path: str = None,
                         config_parameters: dict = {}) -> Tuple:
    """
    Function for loading exllama v2 based model objects.
    :param model_path: Path to model files.
    :param model_file: Model file to load.
        Defaults to None.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param tokenizer_path: Tokenizer path.
        Defaults to None.
    :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param embeddings_path: Embeddings path.
        Defaults to None.
    :param embeddings_parameters: Embeddings loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param config_path: Config path.
        Defaults to None.
    :param config_parameters: Config loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Tuple of config, tokenizer, embeddings, model and generator object.
        Note, that all objects that do not belong to the backend will be None.
    """
    from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer, ExLlamaV2Config
    from exllamav2.generator import ExLlamaV2BaseGenerator

    config = None
    tokenizer = None
    embeddings = None
    model = None
    generator = None

    config = ExLlamaV2Config()
    if config_parameters is not None:
        for key in config_parameters:
            setattr(config, key, config_parameters[key])
    else:
        setattr(config, "model_dir", model_path)
    config.prepare()

    model = ExLlamaV2(config)
    tokenizer = ExLlamaV2Tokenizer(config)
    cache = ExLlamaV2Cache(model)
    model.load_autosplit(cache)
    generator = ExLlamaV2BaseGenerator(
        model, tokenizer, cache)

    return (config, tokenizer, embeddings, model, generator)


def load_langchain_llamacpp_model(model_path: str,
                                  model_file: str = None,
                                  model_parameters: dict = {},
                                  tokenizer_path: str = None,
                                  tokenizer_parameters: dict = {},
                                  embeddings_path: str = None,
                                  embeddings_parameters: dict = {},
                                  config_path: str = None,
                                  config_parameters: dict = {}) -> None:
    """
    Function for loading langchain-llamacpp based model objects.
    :param model_path: Path to model files.
    :param model_file: Model file to load.
        Defaults to None.
    :param model_parameters: Model loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param tokenizer_path: Tokenizer path.
        Defaults to None.
    :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param embeddings_path: Embeddings path.
        Defaults to None.
    :param embeddings_parameters: Embeddings loading kwargs as dictionary.
        Defaults to empty dictionary.
    :param config_path: Config path.
        Defaults to None.
    :param config_parameters: Config loading kwargs as dictionary.
        Defaults to empty dictionary.
    :return: Tuple of config, tokenizer, embeddings, model and generator object.
        Note, that all objects that do not belong to the backend will be None.
    """
    from langchain.llms.llamacpp import LlamaCpp
    from langchain.embeddings.llamacpp import LlamaCppEmbeddings

    config = None
    tokenizer = None
    embeddings = None
    model = None
    generator = None

    if model_file is not None and not model_path.endswith(model_file):
        model_path = os.path.join(model_path, model_file)
    embeddings = LlamaCppEmbeddings(
        model_path=embeddings_path, **embeddings_parameters)
    model = LlamaCpp(model_path=model_path, **model_parameters)
    return (config, tokenizer, embeddings, model, generator)


"""
Abstractions
"""


class ToolArgument(BaseModel):
    """
    Class, representing a tool argument.
    """

    def __init__(self,
                 name: str,
                 type: Type,
                 description: str,
                 value: Any,) -> None:
        """
        Initiation method.
        :param name: Name of the argument.
        :param type: Type of the argument.
        :param description: Description of the argument.
        :param value: Value of the argument.
        """
        self.name = name
        self.type = type
        self.description = description
        self.value = value

    def extract(self, input: str) -> bool:
        """
        Method for extracting argument from input.
        :param input: Input to extract argument from.
        :return: True, if extraction was successful, else False.
        """
        try:
            self.value = self.type(input)
            return True
        except TypeError:
            return False

    def __call__(self) -> str:
        """
        Call method for returning value as string.
        :return: Stored value as string.
        """
        return str(self.value)


class AgentTool(object):
    """
    Class, representing a tool.
    """

    def __init__(self,
                 name: str,
                 description: str,
                 func: Callable,
                 arguments: List[ToolArgument],
                 return_type: Type) -> None:
        """
        Initiation method.
        :param name: Name of the tool.
        :param description: Description of the tool.
        :param func: Function of the tool.
        :param arguments: Arguments of the tool.
        :param return_type: Return type of the tool.
        """
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.return_type = return_type

    def get_guide(self) -> str:
        """
        Method for acquiring the tool guide.
        :return tool guide as string.
        """
        arguments = ", ".join(
            f"{arg.name}: {arg.type}" for arg in self.arguments)
        return f"{self.name}: {self.func.__name__}({arguments}) -> {self.return_type} - {self.description}"

    def __call__(self) -> Any:
        """
        Call method for running tool function with arguments.
        """
        return self.func(**{arg.name: arg.value for arg in self.arguments})


class AgentCache(object):
    """
    Class, representing a message cache.
    """
    supported_backends: List[str] = ["cache"]

    def __init__(self, uuid: str = None, backend: str = "cache", stack: list = None, path: str = None) -> None:
        """
        Initiation method.
        :param uuid: UUID for identifying memory object.
            Defaults to None, in which case a new UUID is generated.
        :param backend: Cache backend. Defaults to "cache".
            Check AgentCache().supported_backends for supported backends.
        :param stack: Stack to initialize cache with.
            Defaults to None.
        :param path: Path for reading and writing stack, if the backend supports it.
            Defaults to None.
        """
        self.stack = None
        self.uuid = uuid4() if uuid is None else uuid
        self.backend = backend
        self.path = path

        self._initiate_stack(stack)

    def _initiate_stack(self, stack: list = None) -> None:
        """
        Method for initiating stack.
        :param stack: Stack initialization.
            Defaults to None.
        """
        pass

    def add(self, message: Tuple[str, str, dict]) -> None:
        """
        Method to add a message to the stack.
        :param message: Message tuple, constisting of agent name, message content and message metadata.
        """
        pass

    def get(self, position: int) -> Tuple[str, str, dict]:
        """
        Method for retrieving message by stack position.
        :param position: Stack position.
        """
        pass


class AgentMemory(object):
    """
    Class, representing a longterm memory.
    """
    supported_backends: List[str] = ["cache"]

    def __init__(self, uuid: str = None, backend: str = "cache", memories: list = None, path: str = None) -> None:
        """
        Initiation method.
        :param uuid: UUID for identifying memory object.
            Defaults to None, in which case a new UUID is generated.
        :param backend: Memory backend. Defaults to "cache".
            Check AgentMemory().supported_backends for supported backends.
        :param memories: Memories to initialize memory with.
            Defaults to None.
        :param path: Path for reading and writing stack, if the backend supports it.
            Defaults to None.
        """
        self.stack = None
        self.uuid = uuid4() if uuid is None else uuid
        self.backend = backend
        self.path = path

        self._initiate_memory(memories)

    def _initiate_memory(self, memories: list = None) -> None:
        """
        Method for initiating memories.
        :param memories: Memory initialization.
            Defaults to None.
        """
        pass

    def add(self, memory: str, metadata: dict) -> None:
        """
        Method to add a memory.
        :param memory: Memory string.
        :param metadata: Metadata for memory.
        """
        pass

    def get(self, filtermasks: List[FilterMask]) -> Tuple[str, str, dict]:
        """
        Method for retrieving message by filtermasks.
        :param filtermasks: List of filtermasks.
        """
        pass

    def retrieve_by_similarity(self, reference: str, filtermasks: List[FilterMask] = None, retrieval_parameters: dict = None) -> Tuple[str, str, dict]:
        """
        Method for retrieving memory by similarity.
        :param reference: Reference for similarity search.
        :param filtermasks: List of filtermasks for additional filering.
        :param retrieval_parameters: Keyword arguments for retrieval.
        """
        pass


class LanguageModelInstance(object):
    """
    Language model class.
    """
    supported_backends: List[str] = ["ctransformers", "transformers",
                                     "llamacpp", "autogptq", "exllamav2", "langchain_llamacpp"]

    def __init__(self,
                 backend: str,
                 model_path: str,
                 model_file: str = None,
                 model_parameters: dict = None,
                 tokenizer_path: str = None,
                 tokenizer_parameters: dict = None,
                 embeddings_path: str = None,
                 embeddings_parameters: dict = None,
                 config_path: str = None,
                 config_parameters: dict = None,
                 default_system_prompt: str = None,
                 use_history: bool = True,
                 history: List[Tuple[str, str, dict]] = None,
                 encoding_parameters: dict = None,
                 embedding_parameters: dict = None,
                 generating_parameters: dict = None,
                 decoding_parameters: dict = None
                 ) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
            Defaults to None.
        :param model_parameters: Model loading kwargs as dictionary.
            Defaults to None.
        :param tokenizer_path: Tokenizer path.
            Defaults to None.
        :param tokenizer_parameters: Tokenizer loading kwargs as dictionary.
            Defaults to None.
        :param embeddings_path: Embeddings path.
            Defaults to None.
        :param embeddings_parameters: Embeddings loading kwargs as dictionary.
            Defaults to None.
        :param config_path: Config path.
            Defaults to None.
        :param config_parameters: Config loading kwargs as dictionary.
            Defaults to None.
        :param default_system_prompt: Default system prompt.
            Defaults to a standard system prompt.
        :param use_history: Flag, declaring whether to use the history.
            Defaults to True.
        :param history: Interaction history as list of (<role>, <message>, <metadata>)-tuples tuples.
            Defaults to None.
        :param encoding_parameters: Kwargs for encoding in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initation methods.
        :param embedding_paramters: Kwargs for embedding as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initation methods.
        :param generating_parameters: Kwargs for generating in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initation methods.
        :param decoding_parameters: Kwargs for decoding in the generation process as dictionary.
            Defaults to None in which case an empty dictionary is created and can be filled depending on the backend in the 
            different initation methods.
        """
        self.backend = backend
        self.system_prompt = "You are a friendly and helpful assistant answering questions based on the context provided." if default_system_prompt is None else default_system_prompt

        self.use_history = use_history
        self.history = [("system", self.system_prompt, {
            "intitated": dt.now()})] if history is None else history

        self.encoding_parameters = {} if encoding_parameters is None else encoding_parameters
        self.embedding_parameters = {} if embedding_parameters is None else embedding_parameters
        self.generating_parameters = {} if generating_parameters is None else generating_parameters
        self.decoding_parameters = {} if decoding_parameters is None else decoding_parameters

        self.config, self.tokenizer, self.embeddings, self.model, self.generator = {
            "ctransformers": load_ctransformers_model,
            "transformers": load_transformers_model,
            "llamacpp": load_llamacpp_model,
            "autogptq": load_autogptq_model,
            "exllamav2": load_exllamav2_model,
            "langchain_llamacpp": load_langchain_llamacpp_model
        }[backend](
            model_path=model_path,
            model_file=model_file,
            model_parameters=model_parameters,
            tokenizer_path=tokenizer_path,
            tokenizer_parameters=tokenizer_parameters,
            embeddings_path=embeddings_path,
            embeddings_parameters=embeddings_parameters,
            config_path=config_path,
            config_parameters=config_parameters
        )

    """
    Generation methods
    """

    def embed(self,
              input: str,
              encoding_parameters: dict = None,
              embedding_parameters: dict = None,
              ) -> List[float]:
        """
        Method for embedding an input.
        :param input: Input to embed.
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param embedding_paramters: Kwargs for embedding as dictionary.
            Defaults to None.
        """
        encoding_parameters = self.encoding_parameters if encoding_parameters is None else encoding_parameters
        embedding_parameters = self.embedding_parameters if embedding_parameters is None else embedding_parameters

        if self.backend == "ctransformers":
            return self.model.embed(input, **embedding_parameters)
        elif self.backend == "langchain_llamacpp":
            # TODO: Implement embeddings for langchain_llamacpp
            pass
        elif self.backend == "transformers":
            input_ids = self.tokenizer.encode(input, **encoding_parameters)
            return self.model.model.embed_tokens(input_ids)
        elif self.backend == "autogptq":
            input_ids = self.tokenizer.encode(input, **encoding_parameters)
            return self.model.model.embed_tokens(input_ids)
        elif self.backend == "llamacpp":
            return self.model.embed(input)
        elif self.backend == "exllamav2":
            input_ids = self.tokenizer.encode(input, **encoding_parameters)
            return self.model.model.embed_tokens(input_ids)

    def generate(self,
                 prompt: str,
                 history_merger: Callable = lambda history: "\n".join(
                     f"<s>{entry[0]}:\n{entry[1]}</s>" for entry in history) + "\n",
                 encoding_parameters: dict = None,
                 generating_parameters: dict = None,
                 decoding_parameters: dict = None) -> Tuple[str, Optional[dict]]:
        """
        Method for generating a response to a given prompt and conversation history.
        :param prompt: Prompt.
        :param history_merger: Merger function for creating full prompt, 
            taking in the prompt history as a list of (<role>, <message>, <metadata>)-tuples as argument (already including the new user prompt).
        :param encoding_parameters: Kwargs for encoding as dictionary.
            Defaults to None.
        :param generating_parameters: Kwargs for generating as dictionary.
            Defaults to None.
        :param decoding_parameters: Kwargs for decoding as dictionary.
            Defaults to None.
        :return: Tuple of textual answer and metadata.
        """
        if not self.use_history:
            self.history = [self.history[0]]
        self.history.append(("user", prompt))
        full_prompt = history_merger(self.history)

        encoding_parameters = self.encoding_parameters if encoding_parameters is None else encoding_parameters
        generating_parameters = self.generating_parameters if generating_parameters is None else generating_parameters
        decoding_parameters = self.decoding_parameters if decoding_parameters is None else decoding_parameters

        metadata = {}
        answer = ""

        start = dt.now()
        if self.backend == "ctransformers" or self.backend == "langchain_llamacpp":
            metadata = self.model(full_prompt, **generating_parameters)
        elif self.backend == "transformers" or self.backend == "autogptq":
            input_tokens = self.tokenizer(
                full_prompt, **encoding_parameters).to(self.model.device)
            output_tokens = self.model.generate(
                **input_tokens, **generating_parameters)[0]
            metadata = self.tokenizer.decode(
                output_tokens, **decoding_parameters)
        elif self.backend == "llamacpp":
            metadata = self.model(full_prompt, **generating_parameters)
            answer = metadata["choices"][0]["text"]
        elif self.backend == "exllamav2":
            metadata = self.generator.generate_simple(
                full_prompt, **generating_parameters)
        self.history.append(("assistant", answer))

        metadata.update({"processing_time": dt.now() -
                        start, "timestamp": dt.now()})

        return answer, metadata


class Agent(object):
    """
    Class, representing an agent.
    """

    def __init__(self,
                 general_llm: LanguageModelInstance,
                 tools: List[AgentTool] = None,
                 memory: AgentMemory = None,
                 cache: AgentCache = None,
                 dedicated_planner_llm: LanguageModelInstance = None,
                 dedicated_actor_llm: LanguageModelInstance = None,
                 dedicated_oberserver_llm: LanguageModelInstance = None) -> None:
        """
        Initiation method.
        :param general_llm: LanguageModelInstance for general tasks.
        :param tools: List of tools to be used by the agent.
            Defaults to None in which case no tools are used.
        :param memory: Memory to use.
            Defaults to None.
        :param cache: Cache to use.
            Defaults to None.
        :param dedicated_planner_llm: LanguageModelInstance for planning.
            Defaults to None in which case the general LLM is used for this task.
        :param dedicated_actor_llm: LanguageModelInstance for acting.
            Defaults to None in which case the general LLM is used for this task.
        :param dedicated_oberserver_llm: LanguageModelInstance for observing.
            Defaults to None in which case the general LLM is used for this task.
        """
        self.general_llm = general_llm
        self.tools = tools
        self.tool_guide = self._create_tool_guide()
        self.cache = cache
        self.memory = memory
        self.planner_llm = self.general_llm if dedicated_planner_llm is None else dedicated_planner_llm
        self.actor_llm = self.general_llm if dedicated_actor_llm is None else dedicated_actor_llm
        self.observer_llm = self.general_llm if dedicated_oberserver_llm is None else dedicated_oberserver_llm

        self.system_prompt = f"""You are a helpful assistant. You have access to the following tools: {self.tool_guide} Your goal is to help the user as best as you can."""

        self.general_llm.use_history = False
        self.general_llm.system_prompt = self.system_prompt

        self.planner_answer_format = f"""Answer in the following format:
            THOUGHT: Formulate precisely what you want to do.
            TOOL: The name of the tool to use. Should be one of [{', '.join(tool.name for tool in self.tools)}]. Only add this line if you want to use a tool in this step.
            INPUTS: The inputs for the tool, separated by a comma. Only add arguments if the tool needs them. Only add the arguments appropriate for the tool. Only add this line if you want to use a tool in this step."""

    def _create_tool_guide(self) -> Optional[str]:
        """
        Method for creating a tool guide.
        """
        if self.tools is None:
            return None
        return "\n\n" + "\n\n".join(tool.get_guide() for tool in self.tools) + "\n\n"

    def loop(self, start_prompt: str) -> Any:
        """
        Method for starting handler loop.
        :param start_prompt: Start prompt.
        :return: Answer.
        """
        self.cache.add(("user", start_prompt, {"timestamp": dt.now()}))
        kickoff_prompt = self.base_prompt + """Which steps need to be taken?
        Answer in the following format:

        STEP 1: Describe the first step. If you want to use a tools, describe how to use it. Use only one tool per step.
        STEP 2: ...
        """
        self.cache.add(("system", kickoff_prompt, {"timestamp": dt.now()}))
        self.cache.add(
            ("general", *self.general_llm.generate(kickoff_prompt)))

        self.system_prompt += f"\n\n The plan is as follows:\n{self.cache.get(-1)[1]}"
        for llm in [self.planner_llm, self.observer_llm]:
            llm.use_history = False
            llm.system_prompt = self.system_prompt
        while not self.cache.get(-1)[1] == "FINISHED":
            for step in [self.plan, self.act, self.observe]:
                step()
                self.report()

    def plan(self) -> Any:
        """
        Method for handling an planning step.
        :return: Answer.
        """
        if self.cache.get(-1)[0] == "general":
            answer, metadata = self.planner_llm.generate(
                f"Plan out STEP 1. {self.planner_answer_format}"
            )
        else:
            answer, metadata = self.planner_llm.generate(
                f"""The current step is {self.cache.get(-1)[1]}
                Plan out this step. {self.planner_answer_format}
                """
            )
        # TODO: Add validation
        self.cache.add("planner", answer, metadata)

    def act(self) -> Any:
        """
        Method for handling an acting step.
        :return: Answer.
        """
        thought = self.cache.get(-1)[1].split("THOUGHT: ")[1].split("\n")[0]
        if "TOOL: " in self.cache.get(-1)[1] and "INPUTS: " in self.cache.get(-1)[1]:
            tool = self.cache.get(-1)[1].split("TOOL: ")[1].split("\n")[0]
            inputs = self.cache.get(-1)[1].split(
                "TOOL: ")[1].split("\n")[0].strip()
            for part in [tool, inputs]:
                if part.endswith("."):
                    part = part[:-1]
                part = part.strip()
            inputs = [inp.strip() for inp in inputs.split(",")]
            # TODO: Catch tool and input failures and repeat previous step.
            tool_to_use = [
                tool_option for tool_option in self.tools if tool.name == tool][0]
            result = tool_to_use.func(
                *[arg.type(inputs[index]) for index, arg in enumerate(tool_to_use.arguments)]
            )
            self.cache.add("actor", f"THOUGHT: {thought}\nRESULT:{result}", {
                "timestamp": dt.now(), "tool_used": tool.name, "arguments_used": inputs})
        else:
            self.cache.add("actor", *self.actor_llm.generate(
                f"Solve the following task: {thought}.\n Answer in following format:\nTHOUGHT: Describe your thoughts on the task.\nRESULT: State your result for the task."
            ))

    def observe(self) -> Any:
        """
        Method for handling an oberservation step.
        :return: Answer.
        """
        current_step = "STEP 1" if self.cache.get(
            -3)[0] == "general" else self.cache.get(-3)[1]
        planner_answer: self.cache.get(-2)[1]
        actor_answer: self.cache.get(-1)[1]
        self.cache.add("observer", *self.observer_llm.generate(
            f"""The current step is {current_step}.
            
            An assistant created the following plan:
            {planner_answer}

            Another assistant implemented this plan as follows:
            {actor_answer}

            Validate, wether the current step is solved. Answer in only one word:
            If the solution is correct and this was the last step, answer 'FINISHED'.
            If the solution is correct but there are more steps, answer 'NEXT'.
            If the solution is not correct, answer the current step in the format 'CURRENT'.
            Your answer should be one of ['FINISHED', 'NEXT', 'CURRENT']
            """
        ))
        # TODO: Add validation and error handling.
        self.cache.add("system", {"FINISHED": "FINISHED", "NEXT": "NEXT", "CURRENT": "CURRENT"}[
            self.cache.get(-1)[1].replace("'", "")], {"timestamp": dt.now()})

    def report(self) -> None:
        """
        Method for printing an report.
        """
        pass


"""
Interfacing
"""


def spawn_language_model_instance(*args: Optional[Any], **kwargs: Optional[Any]) -> Union[LanguageModelInstance, dict]:
    """
    Function for spawning language model instance based on configuration arguments.
    :param args: Arbitrary initiation arguments.
    :param kwargs: Arbitrary initiation keyword arguments.
    :return: Language model instance if configuration was successful else an error report.
    """
    try:
        return LanguageModelInstance(*args, **kwargs)
    except Exception as ex:
        return {"exception": ex, "trace": traceback.format_exc()}


"""
Evaluation and experimentation
"""
TESTING_CONFIGS = {
    #########################
    # llamacpp
    #########################
    "llamacpp_openhermes-2.5-mistral-7b_v2": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Your task is to create a Python script for scraping the first 10 google hits for a given search query. Explain your solution afterwards.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 1024}
        }
    },
    "llamacpp_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 1024}
        }
    },
    "llamacpp_openhermes-2.5-mistral-7b-16k_v2": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Your task is to create a Python script for scraping the first 10 google hits for a given search query. Explain your solution afterwards.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 1024}
        }
    },
    "llamacpp_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 16384},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 2048}
        }
    },
    #########################
    # ctransformers
    #########################
    "ctransformers_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "ctransformers",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_parameters": {"context_length": 4096, "max_new_tokens": 1024},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_new_tokens": 1024}
        }
    },
    "ctransformers_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "ctransformers",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_parameters": {"context_length": 4096, "max_new_tokens": 2048},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_new_tokens": 2048}
        }
    },
    #########################
    # langchain_llamacpp
    #########################
    "langchain_llamacpp_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "langchain_llamacpp",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 1024}
        }
    },
    "langchain_llamacpp_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "langchain_llamacpp",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_parameters": {"n_ctx": 4096},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_parameters": {"max_tokens": 2048}
        }
    },
    #########################
    # autoqptq
    #########################
    "autogptq_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generating_parameters": {"max_new_tokens": 1024},
            "decoding_parameters": {"skip_special_tokens": True}
        }
    },
    "autogptq_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generating_parameters": {"max_new_tokens": 2048},
            "decoding_parameters": {"skip_special_tokens": True}
        }
    },
    "autogptq_openchat_3.5": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_openchat_3.5-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_openchat_3.5-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generating_parameters": {"max_new_tokens": 1024},
            "decoding_parameters": {"skip_special_tokens": True}
        }
    },
}

CURRENTLY_NOT_WORKING = {
    #########################
    # autoqptq
    #########################
    "autogptq_rocket-3b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_rocket-3B-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device_map": "auto", "use_triton": True, "local_files_only": True, "trust_remote_code": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_rocket-3B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generation_parameters": {"max_new_tokens": 128},
            "decoding_parameters": {"skip_special_tokens": True}
        }
    },
    "autogptq_stablecode-instruct-alpha-3b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_stablecode-instruct-alpha-3b-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_parameters": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_stablecode-instruct-alpha-3b-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<s>{entry[0].replace('user', '###Instruction:').replace('assistant', '###Response:')}\n{entry[1]}<|im_end|>" for entry in history if entry[0] != "system") + "\n",
            "tokenizing_parameters": {"return_tensors": "pt"},
            "generation_parameters": {"max_new_tokens": 128},
            "decoding_parameters": {"skip_special_tokens": True, "return_token_type_ids": False}
        }
    },
    #########################
    # exllamav2
    #########################
    "exllamav2_openhermes-2.5-mistral-7b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_parameters": {},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_parameters": {"max_tokens": 1024}
        }
    },
    "exllamav2_openhermes-2.5-mistral-7b-16k": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_parameters": {},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_parameters": {"max_tokens": 2048}
        }
    },
    "exllamav2_rocket-3b": {
        "instance_parameters": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_rocket-3B-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_parameters": {"device_map": "auto", "local_files_only": True, "trust_remote_code": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_rocket-3B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_parameters": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_parameters": {"max_tokens": 128}
        }
    },
}


class Colors:
    std = "\x1b[0;37;40m"
    fat = "\x1b[1;37;40m"
    blk = "\x1b[6;30;42m"
    wrn = "\x1b[31m"
    end = "\x1b[0m"


def run_model_test(configs: List[str] = None) -> dict:
    """
    Function for running model tests based off of configurations.
    :param configs: List of names of configurations to run.
    :return: Answers.
    """
    if configs is None or not all(config in TESTING_CONFIGS for config in configs):
        configs = list(TESTING_CONFIGS.keys())
    answers = {}

    for config in configs:
        try:
            coder = LanguageModelInstance(
                **TESTING_CONFIGS[config]["instance_parameters"]
            )

            answer, metadata = coder.generate(
                **TESTING_CONFIGS[config]["generation_parameters"]
            )
            answers[config] = ("success", answer, metadata)
        except Exception as ex:
            answers[config] = ("failure", ex, traceback.format_exc())
        print(*answers[config])

    for config in configs:
        print("="*100)
        print(
            f"{Colors.blk}Config:{Colors.end}{Colors.fat}{config}{Colors.end}\n")
        if answers[config][0] == "success":
            print(
                f"{Colors.fat}Answer:{Colors.std}\n {answers[config][1]}{Colors.end}\n")
            print(
                f"{Colors.fat}Metadata:{Colors.std}\n {answers[config][2]}{Colors.end}\n")
        else:
            print(
                f"{Colors.wrn}Exception:\n {answers[config][1]}{Colors.end}\n")
            print(
                f"{Colors.wrn}Stacktrace:\n {answers[config][2]}{Colors.end}\n")
        print("="*100)
    return answers
