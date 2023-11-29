
# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import traceback
from typing import List, Tuple, Any, Callable, Optional, Type
from datetime import datetime as dt
from src.configuration import configuration as cfg

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
Abstractions
"""


class ToolArgument(object):
    """
    Class, representing a tool argument.
    """
    name: str
    type: Type
    description: str
    value: Any

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


class Tool(object):
    """
    Class, representing a tool.
    """
    name: str
    description: str
    func: Callable
    arguments: List[ToolArgument]

    def __call__(self) -> Any:
        """
        Call method for running tool function with arguments.
        """
        return self.func(**{arg.name: arg.value for arg in self.arguments})


class LanguageModelInstance(object):
    """
    Language model class.
    """

    def __init__(self,
                 backend: str,
                 model_path: str,
                 model_file: str = None,
                 model_kwargs: dict = None,
                 tokenizer_path: str = None,
                 tokenizer_kwargs: dict = None,
                 config_path: str = None,
                 config_kwargs: dict = None,
                 default_system_prompt: str = "You are a friendly and helpful assistant answering questions based on the context provided.",
                 history: List[Tuple[str, str, dict]] = None
                 ) -> None:
        """
        Initiation method.
        :param backend: Backend for model loading.
        :param model_path: Path to model files.
        :param model_file: Model file to load.
            Defaults to None.
        :param model_kwargs: Model loading kwargs as dictionary.
            Defaults to None.
        :param tokenizer_path: Tokenizer path.
            Defaults to None.
        :param tokenizer_kwargs: Tokenizer loading kwargs as dictionary.
            Defaults to None.
        :param config_path: Config path.
            Defaults to None.
        :param config_kwargs: Config loading kwargs as dictionary.
            Defaults to None.
        :param default_system_prompt: Default system prompt.
            Defaults to a standard system prompt.
        :param history: Interaction history as list of (<role>, <message>, <metadata>)-tuples tuples.
            Defaults to None.
        """
        self.backend = backend
        self.system_prompt = default_system_prompt

        self.config = None
        self.config_path = config_path
        self.config_kwargs = config_kwargs

        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
        self.tokenizer_kwargs = tokenizer_kwargs

        self.model = None
        self.model_path = model_path
        self.model_file = model_file
        self.model_kwargs = model_kwargs

        self.generator = None

        self.history = [{"system", self.system_prompt, {
            "intitated": dt.now()}}] if history is None else history

        initiation_kwargs = {
            "model_file": model_file,
            "model_kwargs": model_kwargs,
            "tokenizer_path": tokenizer_path,
            "tokenizer_kwargs": tokenizer_kwargs,
            "config_path": config_path,
            "config_kwargs": config_kwargs
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

    def _initiate_ctransformers(self, **kwargs) -> None:
        """
        Method for initiating ctransformers based tokenizer and model.
        :param kwargs: Additional arbitrary keyword arguments.
        """
        from ctransformers import AutoConfig as CAutoConfig, AutoModelForCausalLM as CAutoModelForCausalLM, AutoTokenizer as CAutoTokenizer

        self._update_config(CAutoConfig.from_pretrained(
            model_path_or_repo_id=self.model_path), model_kwargs=self.model_kwargs, overwrite_kwargs=True)
        self.model = CAutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=self.model_path, model_file=self.model_file, **self.model_kwargs)
        # TODO: Currently ctransformers' tokenizer from model is not working.
        if False and tokenizer_path is not None:
            if tokenizer_path == model_path:
                self.tokenizer = CAutoTokenizer.from_pretrained(
                    self.model, **self.tokenizer_kwargs)
            else:
                self.tokenizer = CAutoTokenizer.from_pretrained(
                    self.tokenizer_path, **self.tokenizer_kwargs)

    def _initiate_transformers(self, **kwargs) -> None:
        """
        Method for initiating transformers based tokenizer and model.
        :param kwargs: Additional arbitrary keyword arguments.
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self._update_config(AutoConfig.from_pretrained(
            model_path_or_repo_id=self.model_path), model_kwargs=self.model_kwargs, overwrite_kwargs=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_path, **self.tokenizer_kwargs) if self.tokenizer_path is not None else None
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_path, **self.model_kwargs)

    def _initiate_llamacpp(self, **kwargs) -> None:
        """
        Method for initiating llamacpp based tokenizer and model.
        :param kwargs: Additional arbitrary keyword arguments.
        """
        try:
            from llama_cpp_cuda import Llama
        except ImportError:
            from llama_cpp import Llama

        self.model = Llama(model_path=os.path.join(
            self.model_path, self.model_file), **self.model_kwargs)

    def _initiate_autogptq(self, **kwargs) -> None:
        """
        Method for initiating autogptq based tokenizer and model.
        :param kwargs: Additional arbitrary keyword arguments.
        """
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, **self.tokenizer_kwargs) if self.tokenizer_path is not None else None
        self.model = AutoGPTQForCausalLM.from_quantized(
            self.model_path, **self.model_kwargs)

    def _initiate_exllamav2(self, **kwargs) -> None:
        """
        Method for initiating exllamav2 based tokenizer and model.
        :param kwargs: Additional arbitrary keyword arguments.
        """
        from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer, ExLlamaV2Config
        from exllamav2.generator import ExLlamaV2BaseGenerator

        self._update_config(ExLlamaV2Config(),
                            model_kwargs={"config":
                                          {"model_dir": self.model_path}
                                          }, overwrite_kwargs=False)
        self.config.prepare()

        self.model = ExLlamaV2(self.config)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        cache = ExLlamaV2Cache(self.model)
        self.model.load_autosplit(cache)
        self.generator = ExLlamaV2BaseGenerator(
            self.model, self.tokenizer, cache)

    def _initiate_langchain_llamacpp(self, **kwargs) -> None:
        """
        Method for initiating langchain-llamacpp based tokenizer and model.
        :param kwargs: Additional arbitrary keyword arguments.
        """
        from langchain.llms.llamacpp import LlamaCpp

        if self.model_file is not None and not model_path.endswith(self.model_file):
            model_path = os.path.join(model_path, self.model_file)
        self.model = LlamaCpp(model_path=model_path, **self.model_kwargs)

    """
    Generation methods
    """

    def generate(self,
                 prompt: str,
                 history_merger: Callable = lambda history: "\n".join(
                     f"<s>{entry[0]}:\n{entry[1]}</s>" for entry in history) + "\n",
                 encoding_kwargs: dict = None,
                 generating_kwargs: dict = None,
                 decoding_kwargs: dict = None) -> Tuple[str, Optional[dict]]:
        """
        Method for generating a response to a given prompt and conversation history.
        :param prompt: Prompt.
        :param history_merger: Merger function for creating full prompt, 
            taking in the prompt history as a list of (<role>, <message>, <metadata>)-tuples as argument (already including the new user prompt).
        :param encoding_kwargs: Kwargs for encoding as dictionary.
            Defaults to None.
        :param generating_kwargs: Kwargs for generating as dictionary.
            Defaults to None.
        :param decoding_kwargs: Kwargs for decoding as dictionary.
            Defaults to None.
        :return: Tuple of textual answer and metadata.
        """
        self.history.append(("user", prompt))
        full_prompt = history_merger(self.history)

        metadata = None
        answer = None

        if self.backend == "ctransformers" or self.backend == "langchain_llamacpp":
            metadata = self.model(full_prompt, **generating_kwargs)
        elif self.backend == "transformers" or self.backend == "autogptq":
            input_tokens = self.tokenizer(
                full_prompt, **encoding_kwargs).to(self.model.device)
            output_tokens = self.model.generate(
                **input_tokens, **generating_kwargs)[0]
            metadata = self.tokenizer.decode(
                output_tokens, **decoding_kwargs)
        elif self.backend == "llamacpp":
            metadata = self.model(full_prompt, **generating_kwargs)
            answer = metadata["choices"][0]["text"]
        elif self.backend == "exllamav2":
            metadata = self.generator.generate_simple(
                full_prompt, **generating_kwargs)
        self.history.append(("assistant", answer))
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


class Agent(object):
    """
    Class, representing an agent.
    """

    def __init__(self,
                 general_llm: LanguageModelInstance,
                 dedicated_planner_llm: LanguageModelInstance = None,
                 dedicated_actor_llm: LanguageModelInstance = None,
                 dedicated_oberserver_llm: LanguageModelInstance = None) -> None:
        """
        Initiation method.
        :param general_llm: LanguageModelInstance for general tasks.
        :param dedicated_planner_llm: LanguageModelInstance for planning.
            Defaults to None in which case the general LLM is used for this task.
        :param dedicated_actor_llm: LanguageModelInstance for acting.
            Defaults to None in which case the general LLM is used for this task.
        :param dedicated_oberserver_llm: LanguageModelInstance for observing.
            Defaults to None in which case the general LLM is used for this task.
        """
        self.general_llm = general_llm
        self.planner_llm = self.general_llm if dedicated_planner_llm is None else dedicated_planner_llm
        self.actor_llm = self.general_llm if dedicated_actor_llm is None else dedicated_actor_llm
        self.observer_llm = self.general_llm if dedicated_oberserver_llm is None else dedicated_oberserver_llm

    def loop(self, start_prompt: str) -> Any:
        """
        Method for starting handler loop.
        :param start_prompt: Start prompt.
        :return: Answer.
        """
        pass

    def plan(self) -> Any:
        """
        Method for handling an planning step.
        :return: Answer.
        """
        pass

    def act(self) -> Any:
        """
        Method for handling an acting step.
        :return: Answer.
        """
        pass

    def observe(self) -> Any:
        """
        Method for handling an oberservation step.
        :return: Answer.
        """
        pass


"""
Evaluation and experimentation
"""
TESTING_CONFIGS = {
    #########################
    # llamacpp
    #########################
    "llamacpp_openhermes-2.5-mistral-7b": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_kwargs": {"n_ctx": 4096},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_kwargs": {"max_tokens": 1024}
        }
    },
    "llamacpp_openhermes-2.5-mistral-7b-16k": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "llamacpp",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_kwargs": {"n_ctx": 16384},
            "tokenizer_path": "/mnt/Workspaces/Resources/machine_learning_models/text_generation/MODELS/OpenHermes-2.5-Mistral-7B",
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_kwargs": {"max_tokens": 2048}
        }
    },
    #########################
    # ctransformers
    #########################
    "ctransformers_openhermes-2.5-mistral-7b": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "ctransformers",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_kwargs": {"context_length": 4096, "max_new_tokens": 1024},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_kwargs": {"max_new_tokens": 1024}
        }
    },
    "ctransformers_openhermes-2.5-mistral-7b-16k": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "ctransformers",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_kwargs": {"context_length": 4096, "max_new_tokens": 2048},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_kwargs": {"max_new_tokens": 2048}
        }
    },
    #########################
    # langchain_llamacpp
    #########################
    "langchain_llamacpp_openhermes-2.5-mistral-7b": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "backend": "langchain_llamacpp",
            "model_file": "openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            "model_kwargs": {"n_ctx": 4096},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_kwargs": {"max_tokens": 1024}
        }
    },
    "langchain_llamacpp_openhermes-2.5-mistral-7b-16k": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "backend": "langchain_llamacpp",
            "model_file": "openhermes-2.5-mistral-7b-16k.Q4_K_M.gguf",
            "model_kwargs": {"n_ctx": 4096},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GGUF"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generating_kwargs": {"max_tokens": 2048}
        }
    },
    #########################
    # autoqptq
    #########################
    "autogptq_openhermes-2.5-mistral-7b": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_kwargs": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_kwargs": {"return_tensors": "pt"},
            "generating_kwargs": {"max_new_tokens": 1024},
            "decoding_kwargs": {"skip_special_tokens": True}
        }
    },
    "autogptq_openhermes-2.5-mistral-7b-16k": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_kwargs": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_kwargs": {"return_tensors": "pt"},
            "generating_kwargs": {"max_new_tokens": 2048},
            "decoding_kwargs": {"skip_special_tokens": True}
        }
    },
    "autogptq_openchat_3.5": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_openchat_3.5-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_kwargs": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_openchat_3.5-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "tokenizing_kwargs": {"return_tensors": "pt"},
            "generating_kwargs": {"max_new_tokens": 1024},
            "decoding_kwargs": {"skip_special_tokens": True}
        }
    },
}

CURRENTLY_NOT_WORKING = {
    #########################
    # autoqptq
    #########################
    "autogptq_rocket-3b": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_rocket-3B-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_kwargs": {"device_map": "auto", "use_triton": True, "local_files_only": True, "trust_remote_code": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_rocket-3B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "tokenizing_kwargs": {"return_tensors": "pt"},
            "generation_kwargs": {"max_new_tokens": 128},
            "decoding_kwargs": {"skip_special_tokens": True}
        }
    },
    "autogptq_stablecode-instruct-alpha-3b": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_stablecode-instruct-alpha-3b-GPTQ"),
            "backend": "autogptq",
            "model_file": "model.safetensors",
            "model_kwargs": {"device": "cuda:0", "local_files_only": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_stablecode-instruct-alpha-3b-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<s>{entry[0].replace('user', '###Instruction:').replace('assistant', '###Response:')}\n{entry[1]}<|im_end|>" for entry in history if entry[0] != "system") + "\n",
            "tokenizing_kwargs": {"return_tensors": "pt"},
            "generation_kwargs": {"max_new_tokens": 128},
            "decoding_kwargs": {"skip_special_tokens": True, "return_token_type_ids": False}
        }
    },
    #########################
    # exllamav2
    #########################
    "exllamav2_openhermes-2.5-mistral-7b": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_kwargs": {},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_kwargs": {"max_tokens": 1024}
        }
    },
    "exllamav2_openhermes-2.5-mistral-7b-16k": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_kwargs": {},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_OpenHermes-2.5-Mistral-7B-16k-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_kwargs": {"max_tokens": 2048}
        }
    },
    "exllamav2_rocket-3b": {
        "instance_kwargs": {
            "model_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                       "TheBloke_rocket-3B-GPTQ"),
            "backend": "exllamav2",
            "model_file": "model.safetensors",
            "model_kwargs": {"device_map": "auto", "local_files_only": True, "trust_remote_code": True},
            "tokenizer_path": os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                                           "TheBloke_rocket-3B-GPTQ"),
            "default_system_prompt": "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        },
        "generation_kwargs": {
            "prompt": "Create a Python script for scraping the first 10 google hits for a search query.",
            "history_merger": lambda history: "\n".join(f"<|im_start|>{entry[0]}\n{entry[1]}<|im_end|>" for entry in history) + "\n",
            "generation_kwargs": {"max_tokens": 128}
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
                **TESTING_CONFIGS[config]["instance_kwargs"]
            )

            answer, metadata = coder.generate(
                **TESTING_CONFIGS[config]["generation_kwargs"]
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
